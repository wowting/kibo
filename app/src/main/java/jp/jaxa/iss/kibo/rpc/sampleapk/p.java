package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import gov.nasa.arc.astrobee.Result;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import android.graphics.Bitmap;
import android.util.Log;
import org.opencv.core.Mat;
import org.opencv.features2d.SIFT;

public class p extends KiboRpcService {

    private Bitmap inputImage;
    private SIFT sift = SIFT.create();

    private static final String TAG = YourService.class.getSimpleName();

    private static final Point[] AREA_POINTS = {
            new Point(11.0, -10.0, 5.0),
            new Point(10.9, -9.0, 4.0),
            new Point(10.9, -8.0, 4.0),
            new Point(9.9, -6.9, 5.0)
    };

    private static final Quaternion AREA_QUAT = new Quaternion(0f, 0f, -0.707f, 0.707f);

    private static final Point[] OASIS_POINTS = {
            new Point(10.6, -9.7, 4.6),
            new Point(11.0, -8.9, 5.1),
            new Point(10.6, -8.1, 5.1),
            new Point(11.0, -7.2, 4.7)
    };

    @Override
    protected void runPlan1() {
        Log.i(TAG, "=== 測試移動與拍照開始 ===");
        api.startMission();

        try {
            Thread.sleep(2000);
        } catch (InterruptedException ignored) {}

        // 移動到四個辨識區域並拍照
        for (int i = 0; i < AREA_POINTS.length; i++) {
            Result result = api.moveTo(AREA_POINTS[i], AREA_QUAT, true);
            if (!result.hasSucceeded()) {
                Log.w(TAG, "❌ 無法移動到 Area " + (i + 1) + "，原因: " + result.getMessage());
            } else {
                Log.i(TAG, "✅ 成功移動到 Area " + (i + 1));
                try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
                Mat image = api.getMatNavCam();
                if (image != null) {
                    api.saveMatImage(image, "area" + (i + 1) + ".png");
                    Log.i(TAG, "📷 已儲存區域影像 area" + (i + 1));
                } else {
                    Log.w(TAG, "⚠️ 無法取得 NavCam 圖片於 Area " + (i + 1));
                }
            }
        }

        // 移動穿越四個 Oasis
        for (int i = 0; i < OASIS_POINTS.length; i++) {
            Result result = api.moveTo(OASIS_POINTS[i], AREA_QUAT, true);
            if (!result.hasSucceeded()) {
                Log.w(TAG, "❌ 無法穿越 Oasis " + (i + 1) + "，原因: " + result.getMessage());
            } else {
                Log.i(TAG, "🌴 成功穿越 Oasis " + (i + 1));
                try { Thread.sleep(3000); } catch (InterruptedException ignored) {}
            }
        }

        Log.i(TAG, "=== 測試完成 ===");
    }
}
