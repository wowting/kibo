package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import gov.nasa.arc.astrobee.Result;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;

import java.io.InputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class YourService extends KiboRpcService {

    private static final String TAG = YourService.class.getSimpleName();

    private static final String[] TEMPLATE_FILE_NAMES = {
            "coin.png", "compass.png", "coral.png", "crystal.png",
            "diamond.png", "emerald.png", "fossil.png", "key.png",
            "letter.png", "shell.png", "treasure_box.png"
    };

    private static final String[] TEMPLATE_NAMES = {
            "coin", "compass", "coral", "crystal",
            "diamond", "emerald", "fossil", "key",
            "letter", "shell", "treasure_box"
    };
    private static final Point[] AREA_POINTS = {
            new Point(11, -10.0, 5.0),
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

    private static final Point ASTRONAUT_POINT = new Point(11.143, -6.7607, 4.9654);
    private static final Quaternion ASTRONAUT_QUAT = new Quaternion(0f, 0f, 0.707f, 0.707f);

    private int treasureAreaIndex = -1;

    @Override
    protected void runPlan1() {
        Log.i(TAG, "=== 任務開始 ===");
        api.startMission();

        try {
            Thread.sleep(3000);
        } catch (InterruptedException ignored) {}

        String[] areaItems = new String[AREA_POINTS.length];

        for (int i = 0; i < AREA_POINTS.length; i++) {
            api.moveTo(AREA_POINTS[i], AREA_QUAT, true);
            Mat image = api.getMatNavCam();
            if (image != null) {
                api.saveMatImage(image, "area" + (i + 1) + ".png");
                String recognized = recognizeObjectWithTemplates(image);
                areaItems[i] = recognized;
                api.setAreaInfo(i + 1, recognized, recognized.equals("unknown") ? 0 : 1);
                if (recognized.matches("crystal|diamond|emerald")) {
                    treasureAreaIndex = i;
                }
            }
        }

        // 穿越 4 個 Oasis Zone 區域取得加分
        for (Point oasis : OASIS_POINTS) {
            api.moveTo(oasis, AREA_QUAT, true);
            Log.i(TAG, "穿越 Oasis Zone: " + oasis.toString());
        }

        api.moveTo(ASTRONAUT_POINT, ASTRONAUT_QUAT, true);
        api.reportRoundingCompletion();

        Mat targetImage = api.getMatNavCam();
        api.saveMatImage(targetImage, "target.png");
        String[] targetItems = recognizeTwoObjects(targetImage);
        api.notifyRecognitionItem();

        for (int i = 0; i < areaItems.length; i++) {
            if (areaItems[i].equals(targetItems[1])) {
                treasureAreaIndex = i;
                break;
            }
        }

        if (treasureAreaIndex >= 0) {
            api.moveTo(AREA_POINTS[treasureAreaIndex], AREA_QUAT, true);
            api.takeTargetItemSnapshot();
        }
    }



    private String recognizeObjectWithTemplates(Mat image) {
        if (image == null) {
            Log.w(TAG, "NavCam 影像為 null，跳過辨識");
            return "unknown";
        }

        // 轉灰階影像
        Mat grayFull = new Mat();
        Imgproc.cvtColor(image, grayFull, Imgproc.COLOR_RGBA2GRAY);

        // 辨識區 (ROI) 裁切：中央 640x480
        int roiWidth = 640;
        int roiHeight = 480;
        int x = Math.max(0, (grayFull.cols() - roiWidth) / 2);
        int y = Math.max(0, (grayFull.rows() - roiHeight) / 2);
        int w = Math.min(roiWidth, grayFull.cols() - x);
        int h = Math.min(roiHeight, grayFull.rows() - y);
        Rect roi = new Rect(x, y, w, h);
        Mat gray = new Mat(grayFull, roi);

        // 可視化 ROI 框供除錯
        Imgproc.rectangle(grayFull, roi, new Scalar(0, 255, 0), 2);
        api.saveMatImage(grayFull, "debug_roi_marked.png");
        api.saveMatImage(gray, "debug_cropped_roi.png");

        double bestScore = 0.0;
        String bestName = "unknown";

        int attemptLimit = 20; // 最多只進行 20 組比對，避免卡住
        int attemptCount = 0;

        outer:
        for (int t = 0; t < TEMPLATE_FILE_NAMES.length; t++) {
            Mat templateGray = loadTemplateGray(TEMPLATE_FILE_NAMES[t]);
            if (templateGray == null) continue;

            for (int width = 40; width <= 80; width += 20) {
                Mat resized = scalingResizeImage(templateGray, width);
                for (int angle = 0; angle < 360; angle += 60) {
                    if (++attemptCount > attemptLimit) {
                        Log.w(TAG, "已達比對嘗試上限 (" + attemptLimit + ")，跳出比對迴圈");
                        break outer;
                    }

                    Mat rotated = rotateImage(resized, angle);
                    if (rotated == null || gray.empty()) continue;

                    Mat result = new Mat();
                    Imgproc.matchTemplate(gray, rotated, result, Imgproc.TM_CCOEFF_NORMED);
                    Core.MinMaxLocResult mmr = Core.minMaxLoc(result);

                    if (mmr.maxVal > bestScore && mmr.maxVal >= 0.6) {
                        bestScore = mmr.maxVal;
                        bestName = TEMPLATE_NAMES[t];
                    }
                }
            }
        }

        Log.i(TAG, "辨識結果 (限 ROI): " + bestName + "，分數= " + bestScore + ", 總比對次數=" + attemptCount);
        return bestName;
    }






    private String[] recognizeTwoObjects(Mat image) {
        if (image == null) return new String[]{"unknown", "unknown"};
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGBA2GRAY);
        int[] matchCounts = new int[TEMPLATE_FILE_NAMES.length];

        for (int t = 0; t < TEMPLATE_FILE_NAMES.length; t++) {
            Mat template = loadTemplateGray(TEMPLATE_FILE_NAMES[t]);
            if (template == null) continue;
            for (int width = 30; width <= 100; width += 10) {
                Mat resized = scalingResizeImage(template, width);
                if (resized == null) continue;
                for (int angle = 0; angle < 360; angle += 45) {
                    Mat rotated = rotateImage(resized, angle);
                    if (rotated == null) continue;
                    Mat result = new Mat();
                    Imgproc.matchTemplate(gray, rotated, result, Imgproc.TM_CCOEFF_NORMED);
                    double threshold = 0.7;
                    for (int y = 0; y < result.rows(); y++) {
                        for (int x = 0; x < result.cols(); x++) {
                            double[] val = result.get(y, x);
                            if (val != null && val.length > 0 && val[0] >= threshold) {
                                matchCounts[t]++;
                            }
                        }
                    }
                }
            }
        }

        int firstIdx = -1, secondIdx = -1;
        for (int i = 0; i < matchCounts.length; i++) {
            if (firstIdx == -1 || matchCounts[i] > matchCounts[firstIdx]) {
                secondIdx = firstIdx;
                firstIdx = i;
            } else if (secondIdx == -1 || matchCounts[i] > matchCounts[secondIdx]) {
                secondIdx = i;
            }
        }

        String obj1 = (firstIdx >= 0) ? TEMPLATE_NAMES[firstIdx] : "unknown";
        String obj2 = (secondIdx >= 0) ? TEMPLATE_NAMES[secondIdx] : "unknown";
        return new String[]{obj1, obj2};
    }

    private Mat loadTemplateGray(String fileName) {
        try {
            InputStream is = getAssets().open("template/" + fileName);
            Bitmap bmp = BitmapFactory.decodeStream(is);
            Mat mat = new Mat();
            Utils.bitmapToMat(bmp, mat);
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY);
            is.close();
            return mat;
        } catch (IOException e) {
            Log.e(TAG, "無法讀取模板: " + fileName, e);
            return null;
        }
    }

    private Mat scalingResizeImage(Mat image, int width) {
        if (image == null || image.cols() == 0) return null;
        double ratio = (double) width / image.cols();
        int height = (int) (image.rows() * ratio);
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(width, height));
        return resized;
    }

    private Mat rotateImage(Mat img, double angle) {
        if (img == null) return null;
        org.opencv.core.Point center = new org.opencv.core.Point(img.cols() / 2.0, img.rows() / 2.0);
        Mat rotMat = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        Mat dst = new Mat();
        Imgproc.warpAffine(img, dst, rotMat, img.size());
        return dst;
    }


}
