package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.res.AssetManager;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

public class wow_1 extends KiboRpcService {

    private static final String TAG = "KiboRPC";

    private static final Map<Integer, Point> AREA_POINTS = new HashMap<Integer, Point>() {{
        put(1, new Point(11.3, -10.007, 5.287));
        put(2, new Point(11.00, -8.80, 4.518));//3.8
        put(3, new Point(10.95, -7.95, 4.518));//3.8
        put(4, new Point(10.50, -6.90, 4.8));//4.95
    }};

    private static final Map<Integer, Quaternion> AREA_QUATS = new HashMap<Integer, Quaternion>() {{
        put(1, new Quaternion(0f, 0f, -0.707f, 0.707f));
        put(2, new Quaternion(0f, 0.707f, 0f, 0.707f));
        put(3, new Quaternion(0f, 0.707f, 0f, 1f));
        put(4, new Quaternion(0.707f, 0f, 0f, 0.707f));//0.707 0 0 0
    }};

    private static final Point ASTRONAUT_POINT = new Point(11.143, -6.7607, 4.9654);
    private static final Quaternion ASTRONAUT_QUAT = new Quaternion(0f, 0f, 0.707f, 0.707f);

    private final List<String> landmarkItems = Arrays.asList("coin", "compass", "coral", "fossil", "key", "letter", "shell", "treasure_box");
    private final List<String> treasureItems = Arrays.asList("crystal", "diamond", "emerald");
    private final Map<String, Integer> foundItemArea = new HashMap<>();

    @Override
    protected void runPlan1() {
        Log.i(TAG, "Mission started");
        api.startMission();

        visitAndRecognize(1);
        visitAndRecognize(2);
        visitAndRecognize(3);
        visitAndRecognize(4);

        Log.i(TAG, "Moving to astronaut");
        moveToWithRetry(ASTRONAUT_POINT, ASTRONAUT_QUAT);
        api.reportRoundingCompletion();
        Mat astroImg = getImageWithRetry();

        api.saveMatImage(astroImg, "astronaut_target.png");
    ////    String target = recognizeBestMatch(astroImg);
       // Log.i(TAG, "Target item recognized as: " + target);
        api.notifyRecognitionItem();

//       int targetArea = foundItemArea.getOrDefault(target, 4);
//        Point tgtPoint = AREA_POINTS.get(targetArea);
//       Quaternion tgtQuat = AREA_QUATS.get(targetArea);

//       Log.i(TAG, "Moving to target item at area " + targetArea);
//        moveToWithRetry(tgtPoint, tgtQuat);
        api.takeTargetItemSnapshot();
    }

    private void visitAndRecognize(int areaId) {
        if (areaId == 3) {
            Log.i(TAG, "At Oasis area, pausing for 10 seconds...");
            try { Thread.sleep(10000); } catch (InterruptedException e) { Log.w(TAG, "Oasis wait interrupted"); }
        }
        Point pt = AREA_POINTS.get(areaId);
        Quaternion qt = AREA_QUATS.get(areaId);

        Log.i(TAG, "Moving to area " + areaId);
        moveToWithRetry(pt, qt);

        Log.i(TAG, "Capturing image at area " + areaId);
        try { Thread.sleep(2000); } catch (InterruptedException ignored) {}
        Mat img = getImageWithRetry();
        api.saveMatImage(img, "area_" + areaId + ".png");

//        for (String item : landmarkItems) {
//           if (matchTemplate(img, item, 0.8)) {
//               Log.i(TAG, "Matched landmark " + item + " at area " + areaId);
//                api.setAreaInfo(areaId, item, 1);
//               foundItemArea.put(item, areaId);
//            }
//       }
//       for (String item : treasureItems) {
//           if (matchTemplate(img, item, 0.8)) {
//               Log.i(TAG, "Matched treasure " + item + " at area " + areaId);
//                foundItemArea.put(item, areaId);
//           }
//        }
    }

    private Mat getImageWithRetry() {
        Mat image = null;
        for (int i = 0; i < 3; i++) {
            image = api.getMatNavCam();
            if (image != null) return cropWithoutARTag(image);
            try { Thread.sleep(200); } catch (InterruptedException e) {}
        }
        Log.w(TAG, "Failed to get camera image");
        return new Mat();
    }

    private boolean matchTemplate(Mat source, String name, double thresh) {
        Mat templ = loadTemplate(name);
        if (templ.empty()) return false;
        Mat sourceGray = new Mat();
        Imgproc.cvtColor(source, sourceGray, Imgproc.COLOR_BGR2GRAY);
        Mat result = new Mat();
        Imgproc.matchTemplate(sourceGray, templ, result, Imgproc.TM_CCOEFF_NORMED);
        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
        return mmr.maxVal > thresh;
    }

    private String recognizeBestMatch(Mat image) {
        String bestMatch = "";
        double bestScore = -1;
        for (String item : landmarkItems) {
            Mat templ = loadTemplate(item);
            if (templ.empty()) continue;
            Mat sourceGray = new Mat();
            Imgproc.cvtColor(image, sourceGray, Imgproc.COLOR_BGR2GRAY);
            Mat result = new Mat();
            Imgproc.matchTemplate(sourceGray, templ, result, Imgproc.TM_CCOEFF_NORMED);
            Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
            if (mmr.maxVal > bestScore) {
                bestScore = mmr.maxVal;
                bestMatch = item;
            }
        }
        for (String item : treasureItems) {
            Mat templ = loadTemplate(item);
            if (templ.empty()) continue;
            Mat sourceGray = new Mat();
            Imgproc.cvtColor(image, sourceGray, Imgproc.COLOR_BGR2GRAY);
            Mat result = new Mat();
            Imgproc.matchTemplate(sourceGray, templ, result, Imgproc.TM_CCOEFF_NORMED);
            Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
            if (mmr.maxVal > bestScore) {
                bestScore = mmr.maxVal;
                bestMatch = item;
            }
        }
        return bestMatch;
    }

    private Mat loadTemplate(String filename) {
        Mat img = new Mat();
        try {
            AssetManager assetManager = getApplicationContext().getAssets();
            InputStream is = assetManager.open("template/" + filename + ".png");
            byte[] data = new byte[is.available()];
            is.read(data);
            is.close();

            Mat buf = new Mat(1, data.length, CvType.CV_8UC1);
            buf.put(0, 0, data);
            img = Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_GRAYSCALE);
            buf.release();
        } catch (IOException e) {
            Log.e(TAG, "Failed to load template: " + filename, e);
        }
        return img;
    }

    private Mat cropWithoutARTag(Mat original) {
        int width = original.cols();
        int height = original.rows();
        Rect roi = new Rect(0, 0, (int)(width * 0.85), (int)(height * 0.85));
        return new Mat(original, roi);
    }



    private void moveToWithRetry(Point pt, Quaternion qt) {
        api.moveTo(pt, qt, false);
       // for (int i = 0; i < 3; i++) {
         //   Log.i(TAG, "Attempting move to: " + pt.toString());
            //Result result = api.moveTo(pt, qt, false);
//            if (result.hasSucceeded()) {
//                Log.i(TAG, "Move success");
//                break;
//            } else {
//                Log.w(TAG, "Move failed, retrying... " + (i + 1));
//                try { Thread.sleep(300); } catch (InterruptedException ignored) {}
//            }
//        }
    }
}
