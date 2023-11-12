
import jcuda.runtime.cudaDeviceProp;
import org.junit.Ignore;
import org.junit.Test;
import ru.alexander.neuralengine.NeuralEngine;
import ru.alexander.neuralengine.ioformats.NeuralEngineProject;
import ru.alexander.neuralengine.ioformats.NeuralEngineScheme;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;
import static org.junit.Assert.*;

public class InstrumentsTests {
    @Test
    public void testNEPSaveLoad() throws IOException {
        NeuralEngine engine = new NeuralEngine();

        engine.addVariable("in1", 100, 50);
        engine.addVariable("in2", 100, 50);

        engine.compile("""
                add out in1 in2
                mul out out in1
                transpose tin2 in2
                matmul out1 out tin2
                matmul out2 out1 out
                """);
        engine.saveProject("test.nep", new NeuralEngineProject());

        NeuralEngine load = new NeuralEngine();
        load.loadProject("test.nep", new NeuralEngineProject());

        assertEquals(engine, load);
    }

    @Test
    public void testNESSaveLoad() throws IOException {
        NeuralEngine engine = new NeuralEngine();

        engine.addVariable("in1", 100, 50);
        engine.addVariable("in2", 100, 50);

        engine.compile("""
                add out in1 in2
                mul out out in1
                transpose tin2 in2
                matmul out1 out tin2
                matmul out2 out1 out
                """);
        engine.saveProject("test.nes", new NeuralEngineScheme());

        NeuralEngine load = new NeuralEngine();
        load.loadProject("test.nes", new NeuralEngineScheme());

        assertEquals(engine, load);
    }


    @Test
    public void testDevices() {
        int[] nDevices = new int[1];
        cudaGetDeviceCount(nDevices);

        System.out.printf("Number of devices: %d\n", nDevices[0]);

        for (int i = 0; i < nDevices[0]; i++) {
            cudaDeviceProp prop = new cudaDeviceProp();
            cudaGetDeviceProperties(prop, i);
            System.out.println(Arrays.toString(prop.maxGridSize));
            System.out.println(Arrays.toString(prop.maxThreadsDim));
        }
    }



    @Ignore
    @Test
    public void testVisualization() throws IOException {
        NeuralEngine engine = new NeuralEngine();
        System.out.println(engine.getDocumentation());

        engine.addVariable("in1", 100, 50);
        engine.addVariable("in2", 100, 50);

        engine.compile("""
                add out in1 in2
                mul out out in1
                transpose tin2 in2
                matmul out1 out tin2
                matmul out2 out1 out
                """);
        ImageIO.write(engine.visualize(true), "png", new File("test.png"));
    }
    @Ignore
    @Test
    public void testOptimalGridSizeAlgorithms() {
        Random random = new Random();
        boolean optimal = true;
        for (int it = 0; it < 10; it++) {
            int value = random.nextInt(100000);

            int maxN = 0;
            double max = 0;
            for (int i = 1024; i > 1; i--) {
                double curr = (double) (value % i) / i;
                if (curr > max) {
                    max = curr;
                    maxN = i;
                }
            }


            int fastN1 = value / (int) Math.ceil((double) value / 1023) + 1;
            double fastCoef1 = (double) (value % fastN1) / fastN1;

            int fastN2 = (value + 1) / (value / 1024 + 1);
            double fastCoef2 = (double) (value % fastN2) / fastN2;

            System.out.printf("Value: %d \n", value);
            System.out.printf("Ideal Divider: %d Coef: %1.3f \n", maxN, max);

            System.out.printf("Fast 1 Divider: %d Coef: %1.3f \n", fastN1, fastCoef1);
            System.out.printf("Delta 1: %1.3f \n", max - fastCoef1);

            System.out.printf("Fast 2 Divider: %d Coef: %1.3f \n", fastN2, fastCoef2);
            System.out.printf("Delta 2: %1.3f \n", max - fastCoef2);

            System.out.println();
        }
        assertTrue(optimal);
    }
}
