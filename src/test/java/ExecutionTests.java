import junit.framework.TestCase;
import ru.alexander.neuralengine.NeuralEngine;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

public class ExecutionTests extends TestCase {
    private static final int width = 100;
    private static final int height = 100;
    private static final int size = width * height;

    private static final float err = 1e-3f;
    public void testAdd() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
            out[i] = in1[i] + in2[i];
        }

        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                add out in1 in2
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testSub() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
            out[i] = in1[i] - in2[i];
        }

        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                sub out in1 in2
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testMul() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
            out[i] = in1[i] * in2[i];
        }

        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                mul out in1 in2
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testDiv() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
            out[i] = in1[i] / in2[i];
        }

        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                div out in1 in2
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testTranspose() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
            out[(i / width) + height * (i % width)] = in[i];
        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                transpose out in
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testLinearize() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
            out[i] = in[i];
        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                linearize out in
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }

    public void testMatMul() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];

        float[] out = new float[width * width];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
        }

        for (int i = 0; i < width * width; i++) {
            int x = i % width;
            int y = i / width;

            out[i] = 0;
            for (int j = 0; j < height; j++)
                out[i] += in1[j + y * height] * in2[x + j * width];
        }

        engine.addVariable("in1", height, width, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                matmul out in1 in2
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }


    public void testRelu() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
            if (in[i] >= 0) out[i] = in[i] * 0.7f;
            else out[i] = in[i] * 0.03f;
        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                relu out in 0.7 0.03
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testReluDer() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
            if (in[i] >= 0) out[i] = 0.7f;
            else out[i] = 0.03f;
        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                relu_der out in 0.7 0.03
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }

    public void testSigmoid() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
            out[i] = (float) (1.0 / (1 + Math.exp(-0.8 * in[i])));
        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                sigmoid out in 0.8
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testSigmoidDer() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
            double exp = Math.exp(-0.8 * in[i]);
            out[i] = (float) (0.8 * exp / Math.pow(1 + exp, 2));
        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                sigmoid_der out in 0.8
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }

    public void testTangent() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
            out[i] = (float) (2.0 / (1 + Math.exp(-0.4 * in[i])) - 1);
        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                tangent out in 0.4
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testTangentDer() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
            double exp = Math.exp(-0.8 * in[i]);
            out[i] = (float) (1.6 * exp / Math.pow(1 + exp, 2));
        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                tangent_der out in 0.8
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }

    public void testSoftmax() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        float sum = 0;
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
            out[i] = (float) Math.exp(0.6 * in[i]);
            sum += out[i];
        }
        for (int i = 0; i < size; i++)
            out[i] /= sum;

        engine.addVariable("in", width, height, in);

        engine.compile("""
                softmax out in 0.6
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testSoftmaxDer() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        float sum = 0;
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
            out[i] = (float) Math.exp(0.6 * in[i]);
            sum += out[i];
        }
        for (int i = 0; i < size; i++) {
            out[i] = (sum - out[i]) * 0.6f * out[i] / (sum * sum);
        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                softmax_der out in 0.6
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }


    public void testConvEmB() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] mtx = new float[9];

        float[] out = new float[size];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;
        for (int i = 0; i < 9; i++)
            mtx[i] = random.nextFloat() * 2 - 1;

        for (int i = 0; i < size; i++) {
            int x = i % width;
            int y = i / width;

            out[i] = 0;
            for (int dy = 0; dy < 3; dy++) {
                int py = y + dy - 1;
                if (py < 0) continue;
                if (py >= height) continue;

                for (int dx = 0; dx < 3; dx++) {
                    int px = x + dx - 1;
                    if (px < 0) continue;
                    if (px >= width) continue;

                    out[i] += in[px + py * width] * mtx[dx + dy * 3];
                }
            }
        }

        engine.addVariable("in", width, height, in);
        engine.addVariable("mtx", 3, 3, mtx);

        engine.compile("""
                conv out in mtx empty
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testConvExB() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] mtx = new float[9];

        float[] out = new float[size];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;
        for (int i = 0; i < 9; i++)
            mtx[i] = random.nextFloat() * 2 - 1;

        for (int i = 0; i < size; i++) {
            int x = i % width;
            int y = i / width;

            out[i] = 0;
            for (int dy = 0; dy < 3; dy++) {
                int py = Math.max(0, Math.min(height - 1, y + dy - 1));

                for (int dx = 0; dx < 3; dx++) {
                    int px = Math.max(0, Math.min(width - 1, x + dx - 1));

                    out[i] += in[px + py * width] * mtx[dx + dy * 3];
                }
            }
        }

        engine.addVariable("in", width, height, in);
        engine.addVariable("mtx", 3, 3, mtx);

        engine.compile("""
                conv out in mtx extend
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testConvReB() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] mtx = new float[9];

        float[] out = new float[size];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;
        for (int i = 0; i < 9; i++)
            mtx[i] = random.nextFloat() * 2 - 1;

        for (int i = 0; i < size; i++) {
            int x = i % width;
            int y = i / width;

            out[i] = 0;
            for (int dy = 0; dy < 3; dy++) {
                int py = y + dy - 1;
                if (py < 0) py += height;
                if (py >= height) py -= height;

                for (int dx = 0; dx < 3; dx++) {
                    int px = x + dx - 1;
                    if (px < 0) px += width;
                    if (px >= width) px -= width;

                    out[i] += in[px + py * width] * mtx[dx + dy * 3];
                }
            }
        }

        engine.addVariable("in", width, height, in);
        engine.addVariable("mtx", 3, 3, mtx);

        engine.compile("""
                conv out in mtx repeat
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }

    public void testConcatenateVertical() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];

        float[] out = new float[size * 2];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
        }
        System.arraycopy(in1, 0, out, 0, size);
        System.arraycopy(in2, 0, out, size, size);


        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                concatenate out in1 in2 vertical
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testConcatenateHorizontal() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];

        int w2 = width * 2;
        int size2 = size * 2;
        float[] out = new float[size2];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
        }

        for (int i = 0; i < size2; i++) {
            int x = i % w2;
            int y = i / w2;

            if (x < width) out[i] = in1[x + y * width];
            else {
                x -= width;
                out[i] = in2[x + y * width];
            }
        }


        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                concatenate out in1 in2 horizontal
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }

    public void testMinPooling() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        int qSize = size / 4;
        float[] in = new float[size];
        float[] out = new float[qSize];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;

        int hW = width / 2;
        for (int i = 0; i < qSize; i++) {
            out[i] = 1e38f;
            int x = (i % hW) * 2;
            int y = (i / hW) * 2;
            for(int dy = 0; dy < 2; dy++)
                for(int dx = 0; dx < 2; dx++)
                    out[i] = Math.min(out[i], in[(x + dx) + (y + dy) * width]);

        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                pooling out in 2 min
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testMaxPooling() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        int qSize = size / 4;
        float[] in = new float[size];
        float[] out = new float[qSize];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;

        int hW = width / 2;
        for (int i = 0; i < qSize; i++) {
            out[i] = -1e38f;
            int x = (i % hW) * 2;
            int y = (i / hW) * 2;
            for(int dy = 0; dy < 2; dy++)
                for(int dx = 0; dx < 2; dx++)
                    out[i] = Math.max(out[i], in[(x + dx) + (y + dy) * width]);

        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                pooling out in 2 max
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
    public void testAvgPooling() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        int qSize = size / 4;
        float[] in = new float[size];
        float[] out = new float[qSize];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;

        int hW = width / 2;
        for (int i = 0; i < qSize; i++) {
            out[i] = 0;
            int x = (i % hW) * 2;
            int y = (i / hW) * 2;
            for(int dy = 0; dy < 2; dy++)
                for(int dx = 0; dx < 2; dx++)
                    out[i] += in[(x + dx) + (y + dy) * width];
            out[i] /= 4;
        }

        engine.addVariable("in", width, height, in);

        engine.compile("""
                pooling out in 2 avg
                """);

        engine.compute();

        float[] cOut = engine.getVariable("out");
        engine.close();
        assertArrayEquals(out, cOut, err);
    }
}