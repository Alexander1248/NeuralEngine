import jcuda.Pointer;
import junit.framework.TestCase;
import ru.alexander.neuralengine.NeuralEngine;
import ru.alexander.neuralengine.ioformats.NeuralEngineProject;
import ru.alexander.neuralengine.ioformats.NeuralEngineScheme;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

public class InstrumentsTests extends TestCase {
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

}
