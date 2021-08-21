package in.mcxiv.ai.convnet;

import java.awt.image.BufferedImage;

public class VolUtil {

    // img is a DOM element that contains a loaded image
    // returns a Vol of size (W, H, 4). 4 is for RGBA
    // or (W, H, 1) if gray scale
    public static Vol img_to_vol(BufferedImage img, boolean convert_grayscale) {

        var W = img.getWidth();
        var H = img.getHeight();
        int model = (convert_grayscale ? 1 : 4);

        DoubleBuffer data = new DoubleBuffer();
        if (convert_grayscale) {
            for (int i = 0; i < W; i++) {
                for (int j = 0; j < H; j++) {
                    int col = img.getRGB(i, j);
                    double val = /*Red   = */ (col & 0xFF) +
                            /*     Blue  = */ ((col >> 8) & 0xFF) +
                            /*     Green = */ ((col >> 16) & 0xFF);
                    data.add(val / 3);
                }
            }
        } else {
            for (int i = 0; i < W; i++) {
                for (int j = 0; j < H; j++) {
                    int col = img.getRGB(i, j);
                    data.add( /*Red   = */ (col & 0xFF));
                    data.add( /*Blue  = */ ((col >> 8) & 0xFF));
                    data.add( /*Green = */ ((col >> 16) & 0xFF));
                    data.add( /*Alpha = */ (col >> 24));
                }
            }
        }

        for (int i = 0, s = data.size; i < s; i++)
            data.set(i, data.get(i) / 255 - 0.5);

        Vol x = new Vol(W, H, model, 0.0); //input volume (image)
        x.w = data;

        return x;
    }

}
