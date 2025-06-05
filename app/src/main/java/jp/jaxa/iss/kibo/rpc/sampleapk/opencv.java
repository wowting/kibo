package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;


import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

public class opencv extends KiboRpcService {

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "Unable to load OpenCV");
        } else {
            Log.d("OpenCV", "OpenCV loaded successfully");
        }
    }


    protected void onCreate(Bundle savedInstanceState) {


        String basePath = getExternalFilesDir(null).getAbsolutePath();
        String targetPath = basePath + "/astronaut_target.png";
        String templateDirPath = basePath + "/template";

        Mat targetColor = Imgcodecs.imread(targetPath);
        Mat targetGray = new Mat();
        Imgproc.cvtColor(targetColor, targetGray, Imgproc.COLOR_BGR2GRAY);

        // Init SIFT and matcher
        SIFT sift = SIFT.create();
        Mat desTarget = new Mat();
        List<KeyPoint> kpTarget = new ArrayList<>();
        sift.detectAndCompute(targetGray, new Mat(), (MatOfKeyPoint) kpTarget, desTarget);

        BFMatcher matcher = BFMatcher.create(DescriptorMatcher.BRUTEFORCE, false);

        File templateDir = new File(templateDirPath);
        if (!templateDir.exists()) {
            Log.e("SIFT", "Template folder not found.");
            return;
        }

        File[] templateFiles = templateDir.listFiles();
        List<String> detected = new ArrayList<>();

        for (File file : templateFiles) {
            if (!file.getName().endsWith(".png")) continue;

            String name = file.getName().replace(".png", "");
            Mat templateGray = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);

            List<KeyPoint> kpTemplate = new ArrayList<>();
            Mat desTemplate = new Mat();
            sift.detectAndCompute(templateGray, new Mat(), (MatOfKeyPoint) kpTemplate, desTemplate);

            if (desTemplate.empty() || desTarget.empty()) continue;

            List<MatOfDMatch> knnMatches = new ArrayList<>();
            matcher.knnMatch(desTemplate, desTarget, knnMatches, 2);

            List<DMatch> goodMatches = new ArrayList<>();
            for (MatOfDMatch matOfDMatch : knnMatches) {
                DMatch[] matches = matOfDMatch.toArray();
                if (matches.length >= 2 && matches[0].distance < 0.6 * matches[1].distance) {
                    goodMatches.add(matches[0]);
                }
            }

            Log.d("SIFT", name + ": " + goodMatches.size() + " good matches");

            if (goodMatches.size() >= 10) {
                detected.add(name);

                // Draw rectangle at first matched keypoint
                Point pt = kpTarget.get(goodMatches.get(0).trainIdx).pt;
                Point topLeft = new Point(pt.x - 30, pt.y - 30);
                Point bottomRight = new Point(pt.x + 30, pt.y + 30);
                Imgproc.rectangle(targetColor, topLeft, bottomRight, new Scalar(0, 255, 0), 2);
                Imgproc.putText(targetColor, name, new Point(pt.x - 30, pt.y - 35),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 255, 0), 2);
            }
        }

        // Save or show the result
        String outPath = basePath + "/target_sift_detected.jpg";
        Imgcodecs.imwrite(outPath, targetColor);
        Log.d("SIFT", "Detected items: " + detected.toString());
        Log.d("SIFT", "Saved output to: " + outPath);
    }
}
