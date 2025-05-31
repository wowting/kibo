package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.util.Log;
import android.content.res.AssetManager;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Result;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.InputStream;
import java.io.IOException;
import java.util.*;

public class wow extends KiboRpcService {

    private static final String TAG = "KiboRPC";

    private static final Map<Integer, Point> AREA_POINTS = new HashMap<Integer, Point>() {{
        put(1, new Point(10.95, -10.1, 5.2));
        put(2, new Point(11.0, -8.9, 3.76));
        put(3, new Point(11.0, -7.9, 3.76));
        put(4, new Point(9.87, -6.9, 4.9));
    }};

    private static final Map<Integer, Quaternion> AREA_QUATS = new HashMap<Integer, Quaternion>() {{
        put(1, new Quaternion(0f, 0f, -0.707f, 0.707f));
        put(2, fromEuler(0, 90, 0));
        put(3, fromEuler(0, 90, 0));
        put(4, new Quaternion(0f, 0f, 0.707f, 0.707f));
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

        for (int area = 1; area <= 4; area++) {
            Log.i(TAG, "Scanning area " + area);
            visitAndRecognize(area);
        }

        Log.i(TAG, "Moving to astronaut");
        moveToWithRetry(ASTRONAUT_POINT, ASTRONAUT_QUAT);
        api.reportRoundingCompletion();

        Mat astroImg = getImageWithRetry();
        api.saveMatImage(astroImg, "astronaut_target.png");
        String target = recognizeBestMatch(astroImg);
        Log.i(TAG, "Target item recognized as: " + target);
        api.notifyRecognitionItem();

        int targetArea = foundItemArea.getOrDefault(target, 4);
        Point tgtPoint = AREA_POINTS.get(targetArea);
        Quaternion tgtQuat = AREA_QUATS.get(targetArea);

        Log.i(TAG, "Moving to target item at area " + targetArea);
        moveToWithRetry(tgtPoint, tgtQuat);
        api.takeTargetItemSnapshot();
    }

    private void visitAndRecognize(int areaId) {
        Point pt = AREA_POINTS.get(areaId);
        Quaternion qt = AREA_QUATS.get(areaId);

        Log.i(TAG, "Moving to area " + areaId);
        moveToWithRetry(pt, qt);

        Log.i(TAG, "Capturing image at area " + areaId);
        Mat img = getImageWithRetry();
        api.saveMatImage(img, "area_" + areaId + ".png");

        for (String item : landmarkItems) {
            if (matchTemplate(img, item, 0.8)) {
                Log.i(TAG, "Matched landmark " + item + " at area " + areaId);
                api.setAreaInfo(areaId, item, 1);
                foundItemArea.put(item, areaId);
            }
        }
        for (String item : treasureItems) {
            if (matchTemplate(img, item, 0.8)) {
                Log.i(TAG, "Matched treasure " + item + " at area " + areaId);
                foundItemArea.put(item, areaId);
            }
        }
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

    private static Quaternion fromEuler(double roll, double pitch, double yaw) {
        double cy = Math.cos(Math.toRadians(yaw) * 0.5);
        double sy = Math.sin(Math.toRadians(yaw) * 0.5);
        double cp = Math.cos(Math.toRadians(pitch) * 0.5);
        double sp = Math.sin(Math.toRadians(pitch) * 0.5);
        double cr = Math.cos(Math.toRadians(roll) * 0.5);
        double sr = Math.sin(Math.toRadians(roll) * 0.5);
        double w = cr * cp * cy + sr * sp * sy;
        double x = sr * cp * cy - cr * sp * sy;
        double y = cr * sp * cy + sr * cp * sy;
        double z = cr * cp * sy - sr * sp * cy;
        return new Quaternion((float) x, (float) y, (float) z, (float) w);
    }

    private void moveToWithRetry(Point pt, Quaternion qt) {
        for (int i = 0; i < 3; i++) {
            Log.i(TAG, "Attempting move to: " + pt.toString());
            Result result = api.moveTo(pt, qt, false);
            if (result.hasSucceeded()) {
                Log.i(TAG, "Move success");
                break;
            } else {
                Log.w(TAG, "Move failed, retrying... " + (i + 1));
                try { Thread.sleep(300); } catch (InterruptedException ignored) {}
            }
        }
    }
}
