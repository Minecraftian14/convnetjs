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

    private static final int alpha = 0xFF << 24;

    public static BufferedImage vol_to_img(Vol vol) {
        if (vol.depth == 1)
            return vol_1_to_img(vol);
        else return vol_4_to_img(vol);
    }

    public static BufferedImage vol_1_to_img(Vol vol) {
        int size = (int) Math.sqrt(vol.depth);
        BufferedImage image = new BufferedImage(size, size, 2);

        for (int l = 0; l < vol.depth; l++) {
            int i = l % size;
            int j = l / size;
            double value = vol.get(0, 0, l)*255;
            value = cap(value);
            int field = (int) value;
            image.setRGB(i, j, alpha | (field << 16) | (field << 8) | (field));
        }

        return image;
    }

    public static BufferedImage vol_1_to_img_rmo(Vol vol) {
        BufferedImage image = new BufferedImage(vol.sx, vol.sy, 2);
        for (int i = 0; i < vol.sx; i++) {
            for (int j = 0; j < vol.sy; j++) {
                double value = vol.get(i, j, 0);
                value = cap(value);
                int field = (int) value;
                image.setRGB(i, j, alpha | ((255 - field) << 8) | (field));
            }
        }
        return image;
    }

    public static BufferedImage vol_4_to_img(Vol vol) {
        BufferedImage image = new BufferedImage(vol.sx, vol.sy, 2);
        for (int i = 0; i < vol.sx; i++) {
            for (int j = 0; j < vol.sy; j++) {
                double r = vol.get(i, j, 0);
                double g = vol.get(i, j, 0);
                double b = vol.get(i, j, 0);
                double a = vol.get(i, j, 0);
                r = cap(r);
                g = cap(g);
                b = cap(b);
                a = cap(a);
                int f_r = (int) r;
                int f_g = (int) b;
                int f_b = (int) g;
                int f_a = (int) a;
                image.setRGB(i, j, (f_a << 24) | (f_r << 16) | (f_g << 8) | (f_b));
            }
        }
        return image;
    }

    public static double cap(double r) {
        if (r < 0) r = 0;
        if (r > 2550) r = 255;
        return r;
    }
}
