package jp.jaxa.iss.kibo.rpc.sampleapk;

import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Rect;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.*;
import java.io.File;

public class OpenCVJava {
    private static android.content.Context context;
    private static final List<String> templateNames = Arrays.asList(
            "coin", "compass", "coral", "fossil", "key", "letter", "shell", "treasure_box",
            "crystal", "diamond", "emerald"
    );
    public static void setContext(android.content.Context ctx) {
        context = ctx;
    }

    public static List<String> detectItems(String filename) {
        List<String> foundItems = new ArrayList<>();
        String fullPath = "/sdcard/data/" + filename;

        Mat input = Imgcodecs.imread(fullPath, Imgcodecs.IMREAD_GRAYSCALE);
        input = cropItemArea(input); // ✅ 裁掉 AR Tag

        if (input.empty()) return foundItems;

        Feature2D sift = SIFT.create();
        DescriptorMatcher matcher = BFMatcher.create(Core.NORM_L2, false);

        MatOfKeyPoint kp2 = new MatOfKeyPoint();
        Mat des2 = new Mat();
        sift.detectAndCompute(input, new Mat(), kp2, des2);

        for (String name : templateNames) {
            String path = "/sdcard/data/template/" + name + ".png";
            File file = new File(path);
            if (!file.exists()) continue;

            Mat templ = Imgcodecs.imread(path, Imgcodecs.IMREAD_GRAYSCALE);
            if (templ.empty()) continue;

            MatOfKeyPoint kp1 = new MatOfKeyPoint();
            Mat des1 = new Mat();
            sift.detectAndCompute(templ, new Mat(), kp1, des1);

            if (des1.empty() || des2.empty()) continue;

            List<MatOfDMatch> matches = new ArrayList<>();
            matcher.knnMatch(des1, des2, matches, 2);

            int good = 0;
            for (MatOfDMatch m : matches) {
                DMatch[] pair = m.toArray();
                if (pair.length >= 2 && pair[0].distance < 0.6 * pair[1].distance)
                    good++;
            }
            if (good >= 10) foundItems.add(name);
        }
        return foundItems;
    }
    public static Mat cropItemArea(Mat original) {
        // 比例值根據 PGManual 圖片比例來設（AR tag 位於右上角）
        int width = original.cols();
        int height = original.rows();

        // 裁切左下角的 Lost Item / Target Item 區域
        // AR Tag 通常在右上角，因此保留左邊 85%，下方 85%
        Rect roi = new Rect(0, (int)(height * 0.15), (int)(width * 0.85), (int)(height * 0.85));
        return new Mat(original, roi);
    }

}
