import ru.alexander.neuralengine.NeuralEngine;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

public class VisualizationTests {
    public static void main(String[] args) throws IOException {
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
}
