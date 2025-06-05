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
        put(1, new Point(11.3, -10.007, 5.287));
      //  put(1, new Point(11.3, -10.007, 5.287));
        put(2, new Point(10.727, -8.80, 4.523));//X:11
        put(3, new Point(10.95, -7.75, 4.523));//3.8   -7.623
        put(4, new Point(10.50, -6.573, 4.8));//4.95
    }};

    private static final Map<Integer, Quaternion> AREA_QUATS = new HashMap<Integer, Quaternion>() {{
        put(1, new Quaternion(0f, 0f, -0.707f, 0.707f));
        put(2, new Quaternion(0f, 0.707f, 0f, 1f));
        put(3, new Quaternion(0f, 0.707f, 0f, 1f));
        put(4, new Quaternion(0f, 0.707f, 0.707f, 0f));//0.707 0 0 0   0.707 0 0 0.707 0.707 0.707
    }};

    private static final Map<Integer, Point> AREA_POINTS2 = new HashMap<Integer, Point>() {{
        put(1, new Point(11.3, -8.8, 5.287));
        //  put(1, new Point(11.3, -10.007, 5.287));
        put(2, new Point(11.3, -8.80, 4.523));//X:11
        put(3, new Point(10.95, -7.4, 4.523));//3.8   -7.623
        put(4, new Point(10.50, -6.90, 4.8));//4.95
    }};

    private static final Map<Integer, Quaternion> AREA_QUATS2 = new HashMap<Integer, Quaternion>() {{
        put(1, new Quaternion(0f, 0f, -0.707f, 0.707f));
        put(2, new Quaternion(0f, 0.707f, 0f, 1f));
        put(3, new Quaternion(0f, 0.707f, 0f, 1f));
        put(4, new Quaternion(-0.707f, 0.707f, 0f, 0f));//0.707 0 0 0   0.707 0 0 0.707
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
        try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
        api.moveTo(AREA_POINTS2.get(1), AREA_QUATS2.get(1), false);
        try { Thread.sleep(4000); } catch (InterruptedException ignored) {}
        visitAndRecognize(2);
        try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
        visitAndRecognize(3);
        try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
        visitAndRecognize(4);

        Log.i(TAG, "Moving to astronaut");
        moveToWithRetry(ASTRONAUT_POINT, ASTRONAUT_QUAT);
        api.reportRoundingCompletion();

        try { Thread.sleep(2000); } catch (InterruptedException ignored) {}

        Mat astroImg = getImageWithRetry();

        api.saveMatImage(astroImg, "astronaut_target.png");

        api.notifyRecognitionItem();

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


    private Mat cropWithoutARTag(Mat original) {
        int width = original.cols();
        int height = original.rows();
        Rect roi = new Rect(0, 0, (int)(width * 0.85), (int)(height * 0.85));
        return new Mat(original, roi);
    }



    private void moveToWithRetry(Point pt, Quaternion qt) {
        api.moveTo(pt, qt, false);

    }
}
