package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.util.Log;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Result;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import org.opencv.core.Mat;

import java.util.*;

public class ww extends KiboRpcService {
    private static final String TAG = "KiboRPC";

    private static final Map<Integer, Point> AREA_POINTS = new HashMap<Integer, Point>() {{
        put(1, new Point(11.3, -10.007, 5.287));
        put(2, new Point(11.00, -8.80, 4.518));
        put(3, new Point(10.95, -7.95, 4.518));
        put(4, new Point(10.50, -6.90, 4.8));
    }};

    private static final Map<Integer, Quaternion> AREA_QUATS = new HashMap<Integer, Quaternion>() {{
        put(1, new Quaternion(0f, 0f, -0.707f, 0.707f));
        put(2, new Quaternion(0f, 0.707f, 0f, 0.707f));
        put(3, new Quaternion(0f, 0.707f, 0f, 0.707f));
        put(4, new Quaternion(0.707f, 0f, 0f, 0.707f));
    }};

    private static final Point ASTRONAUT_POINT = new Point(11.143, -6.7607, 4.9654);
    private static final Quaternion ASTRONAUT_QUAT = new Quaternion(0f, 0f, 0.707f, 0.707f);

    @Override
    protected void runPlan1() {
        copyTemplatesFromAssets();
        Log.i(TAG, "Mission started");
        api.startMission();

        for (int i = 1; i <= 4; i++) {
            visitAndRecognize(i);
        }

        Log.i(TAG, "Moving to astronaut");
        moveToWithRetry(ASTRONAUT_POINT, ASTRONAUT_QUAT);
        api.reportRoundingCompletion();

        try { Thread.sleep(2000); } catch (InterruptedException ignored) {}

        Mat astroImg = getImageWithRetry();
        api.saveMatImage(astroImg, "astronaut_target.png");

        List<String> targetItems = OpenCVJava.detectItems("astronaut_target.png");
        Log.i(TAG, "Target Items: " + targetItems);
        api.notifyRecognitionItem();

        // 比對每張 area 圖
        Map<Integer, List<String>> areaItemMap = new HashMap<>();
        for (int i = 1; i <= 4; i++) {
            List<String> detected = OpenCVJava.detectItems("area_" + i + ".png");
            areaItemMap.put(i, detected);
        }

        // 逐區找出包含任一目標 item 的區域並拍照
        for (int i = 1; i <= 4; i++) {
            List<String> found = areaItemMap.get(i);
            if (found != null && !Collections.disjoint(targetItems, found)) {
                Log.i(TAG, "🛰️ Moving to Area " + i + " for target match");
                moveToWithRetry(AREA_POINTS.get(i), AREA_QUATS.get(i));
                api.takeTargetItemSnapshot();
            }
        }
    }

    private void visitAndRecognize(int areaId) {
        Point pt = AREA_POINTS.get(areaId);
        Quaternion qt = AREA_QUATS.get(areaId);
        moveToWithRetry(pt, qt);
        try { Thread.sleep(2000); } catch (InterruptedException ignored) {}
        Mat img = getImageWithRetry();
        api.saveMatImage(img, "area_" + areaId + ".png");
    }

    private Mat getImageWithRetry() {
        Mat image = null;
        for (int i = 0; i < 3; i++) {
            image = api.getMatNavCam();
            if (image != null && !image.empty()) return image;
            try { Thread.sleep(200); } catch (InterruptedException ignored) {}
        }
        return new Mat();
    }

    private void moveToWithRetry(Point pt, Quaternion qt) {
        for (int i = 0; i < 3; i++) {
            Result result = api.moveTo(pt, qt, false);
            if (result.hasSucceeded()) break;
            try { Thread.sleep(300); } catch (InterruptedException ignored) {}
        }
    }

    private void copyTemplatesFromAssets() {
        try {
            String targetDir = "/sdcard/data/template/";
            java.io.File dir = new java.io.File(targetDir);
            if (!dir.exists()) dir.mkdirs();

            android.content.Context context = getApplicationContext(); // ✅ 使用這個
            android.content.res.AssetManager am = context.getAssets();
            String[] templates = am.list("template");

            for (String filename : templates) {
                java.io.InputStream is = am.open("template/" + filename);
                java.io.File outFile = new java.io.File(dir, filename);
                java.io.FileOutputStream os = new java.io.FileOutputStream(outFile);

                byte[] buffer = new byte[1024];
                int length;
                while ((length = is.read(buffer)) > 0) {
                    os.write(buffer, 0, length);
                }
                is.close();
                os.close();
                Log.i(TAG, "✔ Copied: " + filename);
            }
        } catch (Exception e) {
            Log.e(TAG, "❌ Failed to copy templates: " + e.getMessage());
        }
    }


}
