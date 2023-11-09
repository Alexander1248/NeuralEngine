package ru.alexander;

import ru.alexander.neuralengine.NeuralEngine;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

public class App {
    public static void main(String[] args) throws IOException {
        NeuralEngine engine = new NeuralEngine();
        engine.addVariable("in", 128, 128);
        engine.addVariable("cw1", 5, 5);
        engine.addVariable("cw2", 5, 5);
        engine.addVariable("cw3", 5, 5);

        engine.addVariable("wV1", 128, 256);
        engine.addVariable("wV2", 64, 128);

        engine.addVariable("wE1", 192, 256);
        engine.addVariable("wE2", 128, 192);


        engine.compile("""
                //            Convolution
                conv c1 in cw1 extend
                relu r1 c1 1 0.01
                pooling p1 r1 2 max
                
                conv c2 p1 cw2 extend
                relu r2 c2 1 0.01
                pooling p2 r2 2 max
                
                conv c3 p2 cw3 extend
                relu r3 c3 1 0.01
                pooling p3 r3 2 max
                
                linearize l p3
                
                //            Vector feed forward
                matmul sumV1 l wV1
                sigmoid hV1 sumV1 1
                
                matmul sumV2 hV1 wV2
                sigmoid hV2 sumV2 1
                
                
                //            Emotion feed forward
                matmul sumE1 l wE1
                sigmoid hE1 sumE1 1
                
                matmul sumE2 hE1 wE2
                sigmoid hE2 sumE2 1
                
                
                """);

        ImageIO.write(engine.visualize(true), "png", new File("test.png"));
    }
}
