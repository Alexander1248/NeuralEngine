import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import org.junit.rules.TestWatcher;
import org.junit.runner.Description;
import ru.alexander.neuralengine.NeuralEngine;

import java.io.IOException;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

public class ExecutionTests {
    private static final long timeout = 30000;


    private static final int width = 1000;
    private static final int height = 1000;
    private static final int size = width * height;

    private static final float err = 1e-3f;

    private static final double tScale = 1e-6;


    @Rule
    public TestRule watcher = new TestWatcher() {
        protected void starting(Description description) {
            System.out.println();
            System.out.println("Test: " + description.getMethodName().substring(4));
        }
    };

    @Test(timeout = timeout)
    public void testAdd() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];
        float[] out = new float[size];

        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
        }
        long t = System.nanoTime();
        for (int i = 0; i < size; i++)
            out[i] = in1[i] + in2[i];
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                add out in1 in2
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testSub() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
        }
        long t = System.nanoTime();
        for (int i = 0; i < size; i++)
            out[i] = in1[i] - in2[i];
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                sub out in1 in2
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testMul() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
        }
        long t = System.nanoTime();
        for (int i = 0; i < size; i++)
            out[i] = in1[i] * in2[i];
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                mul out in1 in2
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testDiv() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
        }
        long t = System.nanoTime();
        for (int i = 0; i < size; i++)
            out[i] = in1[i] / in2[i];
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                div out in1 in2
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testTranspose() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;

        long t = System.nanoTime();
        for (int i = 0; i < size; i++)
            out[(i / width) + height * (i % width)] = in[i];
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);
        engine.addVariable("in", width, height, in);

        engine.compile("""
                transpose out in
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testLinearize() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }
        long t = System.nanoTime();
        System.arraycopy(in, 0, out, 0, size);
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);

        engine.compile("""
                linearize out in
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }


//    @Ignore
    @Test(timeout = timeout)
    public void testMatMul1() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];
        float[] out = new float[width * width];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        for (int i = 0; i < width * width; i++) {
            int x = i % width;
            int y = i / width;

            out[i] = 0;
            for (int j = 0; j < height; j++)
                out[i] += in1[j + y * height] * in2[x + j * width];
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in1", height, width, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                matmul out in1 in2
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
//    @Ignore
    @Test(timeout = timeout)
    public void testMatMul2() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];
        float[] out = new float[height * height];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        for (int i = 0; i < height * height; i++) {
            int x = i % height;
            int y = i / height;

            out[i] = 0;
            for (int j = 0; j < width; j++)
                out[i] += in1[j + y * width] * in2[x + j * height];
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", height, width, in2);

        engine.compile("""
                matmul out in1 in2
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }


    @Test(timeout = timeout)
    public void testRelu() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
            if (in[i] >= 0) out[i] = in[i] * 0.7f;
            else out[i] = in[i] * 0.03f;
        }
        long t = System.nanoTime();
        for (int i = 0; i < size; i++) {
            if (in[i] >= 0) out[i] = in[i] * 0.7f;
            else out[i] = in[i] * 0.03f;
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);

        engine.compile("""
                relu out in 0.7 0.03
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testReluDer() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }
        long t = System.nanoTime();
        for (int i = 0; i < size; i++) {
            if (in[i] >= 0) out[i] = 0.7f;
            else out[i] = 0.03f;
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);

        engine.compile("""
                relu_der out in 0.7 0.03
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }

    @Test(timeout = timeout)
    public void testSigmoid() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }
        long t = System.nanoTime();
        for (int i = 0; i < size; i++) {
            out[i] = (float) (1.0 / (1 + Math.exp(-0.8 * in[i])));
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);

        engine.compile("""
                sigmoid out in 0.8
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testSigmoidDer() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        for (int i = 0; i < size; i++) {
            double exp = Math.exp(-0.8 * in[i]);
            out[i] = (float) (0.8 * exp / Math.pow(1 + exp, 2));
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);

        engine.compile("""
                sigmoid_der out in 0.8
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }

    @Test(timeout = timeout)
    public void testTangent() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        for (int i = 0; i < size; i++) {
            out[i] = (float) (2.0 / (1 + Math.exp(-0.4 * in[i])) - 1);
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);

        engine.compile("""
                tangent out in 0.4
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testTangentDer() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        for (int i = 0; i < size; i++) {
            double exp = Math.exp(-0.8 * in[i]);
            out[i] = (float) (1.6 * exp / Math.pow(1 + exp, 2));
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);

        engine.compile("""
                tangent_der out in 0.8
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }

    @Test(timeout = timeout)
    public void testSoftmax() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        float[] sum = new float[height];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        for (int i = 0; i < size; i++) {
            out[i] = (float) Math.exp(0.6 * in[i]);
            sum[i / width] += out[i];
        }
        for (int i = 0; i < size; i++)
            out[i] /= sum[i / width];
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);

        engine.compile("""
                softmax out in 0.6
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testSoftmaxDer() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        float[] sum = new float[height];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        for (int i = 0; i < size; i++) {
            out[i] = (float) Math.exp(0.6 * in[i]);
            sum[i / width] += out[i];
        }
        for (int i = 0; i < size; i++) {
            int y = i / width;
            out[i] = (sum[y] - out[i]) * 0.6f * out[i] / (sum[y] * sum[y]);
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);
        engine.addVariable("in", width, height, in);

        engine.compile("""
                softmax_der out in 0.6
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }


    @Test(timeout = timeout)
    public void testConvEmB() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] mtx = new float[9];

        float[] out = new float[size];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;
        for (int i = 0; i < 9; i++)
            mtx[i] = random.nextFloat() * 2 - 1;


        long t = System.nanoTime();
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
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);
        engine.addVariable("mtx", 3, 3, mtx);

        engine.compile("""
                conv out in mtx empty
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testConvExB() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] mtx = new float[9];

        float[] out = new float[size];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;
        for (int i = 0; i < 9; i++)
            mtx[i] = random.nextFloat() * 2 - 1;

        long t = System.nanoTime();
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
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);
        engine.addVariable("mtx", 3, 3, mtx);

        engine.compile("""
                conv out in mtx extend
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testConvReB() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] mtx = new float[9];

        float[] out = new float[size];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;
        for (int i = 0; i < 9; i++)
            mtx[i] = random.nextFloat() * 2 - 1;

        long t = System.nanoTime();
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
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);
        engine.addVariable("mtx", 3, 3, mtx);

        engine.compile("""
                conv out in mtx repeat
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }

    @Test(timeout = timeout)
    public void testConcatenateVertical() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];

        float[] out = new float[size * 2];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        System.arraycopy(in1, 0, out, 0, size);
        System.arraycopy(in2, 0, out, size, size);
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);


        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                concatenate out in1 in2 vertical
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testConcatenateHorizontal() throws IOException {
        NeuralEngine.initCuda();
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

        long t = System.nanoTime();
        for (int i = 0; i < size2; i++) {
            int x = i % w2;
            int y = i / w2;

            if (x < width) out[i] = in1[x + y * width];
            else {
                x -= width;
                out[i] = in2[x + y * width];
            }
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);


        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                concatenate out in1 in2 horizontal
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }

    @Test(timeout = timeout)
    public void testMinPooling() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        int qSize = size / 4;
        float[] in = new float[size];
        float[] out = new float[qSize];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;

        long t = System.nanoTime();
        int hW = width / 2;
        for (int i = 0; i < qSize; i++) {
            out[i] = 1e38f;
            int x = (i % hW) * 2;
            int y = (i / hW) * 2;
            for(int dy = 0; dy < 2; dy++)
                for(int dx = 0; dx < 2; dx++)
                    out[i] = Math.min(out[i], in[(x + dx) + (y + dy) * width]);

        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);

        engine.compile("""
                pooling out in 2 min
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testMaxPooling() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        int qSize = size / 4;
        float[] in = new float[size];
        float[] out = new float[qSize];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;

        long t = System.nanoTime();
        int hW = width / 2;
        for (int i = 0; i < qSize; i++) {
            out[i] = -1e38f;
            int x = (i % hW) * 2;
            int y = (i / hW) * 2;
            for(int dy = 0; dy < 2; dy++)
                for(int dx = 0; dx < 2; dx++)
                    out[i] = Math.max(out[i], in[(x + dx) + (y + dy) * width]);

        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);

        engine.compile("""
                pooling out in 2 max
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
    @Test(timeout = timeout)
    public void testAvgPooling() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        int qSize = size / 4;
        float[] in = new float[size];
        float[] out = new float[qSize];
        for (int i = 0; i < size; i++)
            in[i] = random.nextFloat() * 2 - 1;

        long t = System.nanoTime();
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
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in", width, height, in);

        engine.compile("""
                pooling out in 2 avg
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }

    @Test(timeout = timeout)
    public void testSum() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[1];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        for (int i = 0; i < size; i++)
            out[0] += in[i];
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);
        engine.addVariable("in", width, height, in);

        engine.compile("""
                sum out in
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, 0.1f);
    }

    @Test(timeout = timeout)
    public void testFlipX() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        for (int i = 0; i < size; i++)
            out[i] = in[(width - 1 - i % width) + (i / width) * width];
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);
        engine.addVariable("in", width, height, in);

        engine.compile("""
                flip out in x
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, 0.1f);
    }
    @Test(timeout = timeout)
    public void testFlipY() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        for (int i = 0; i < size; i++)
            out[i] = in[i % width + (height - 1 - i / width) * width];
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);
        engine.addVariable("in", width, height, in);

        engine.compile("""
                flip out in y
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, 0.1f);
    }
    @Test(timeout = timeout)
    public void testSwapColumns() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        System.arraycopy(in, 0, out, 0, size);
        for (int i = 0; i < height; i++) {
            out[3 + i * width] = in[507 + i * width];
            out[507 + i * width] = in[3 + i * width];
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);
        engine.addVariable("in", width, height, in);

        engine.compile("""
                swap out in 3 507 columns
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, 0.1f);
    }
    @Test(timeout = timeout)
    public void testSwapRows() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in = new float[size];
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        System.arraycopy(in, 0, out, 0, size);
        for (int i = 0; i < height; i++) {
            out[i + 17 * width] = in[i + 214 * width];
            out[i + 214 * width] = in[i + 17 * width];
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);
        engine.addVariable("in", width, height, in);

        engine.compile("""
                swap out in 17 214 rows
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, 0.1f);
    }

    @Test(timeout = timeout)
    public void testScript() throws IOException {
        NeuralEngine.initCuda();
        NeuralEngine engine = new NeuralEngine();
        Random random = new Random();

        float[] in1 = new float[size];
        float[] in2 = new float[size];

        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            in1[i] = random.nextFloat() * 2 - 1;
            in2[i] = random.nextFloat() * 2 - 1;
        }

        long t = System.nanoTime();
        for (int i = 0; i < size; i++) {
            out[i] = (in1[i] + in2[i]) * (in1[i] - in2[i]);
            out[i] = (float) (1.0 / (1 + Math.exp(-0.637 * out[i])));
        }
        System.out.printf("CPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        engine.addVariable("in1", width, height, in1);
        engine.addVariable("in2", width, height, in2);

        engine.compile("""
                add sum in1 in2
                sub delta in1 in2
                mul out sum delta
                sigmoid out out 0.637
                """);

        t = System.nanoTime();
        engine.compute();
        System.out.printf("GPU Execution time: %1.7f \n", (double) (System.nanoTime() - t) * tScale);

        float[] cOut = engine.getVariable("out");
        engine.close();
        NeuralEngine.closeCuda();
        assertArrayEquals(out, cOut, err);
    }
}
