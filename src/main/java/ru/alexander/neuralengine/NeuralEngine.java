package ru.alexander.neuralengine;

import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.instructions.*;

import java.io.IOException;

public class NeuralEngine extends GpuExecutor {

    //    nvcc -arch=all-major -fatbin matrices.cu -o matrices.fatbin
    public NeuralEngine() throws IOException {
        super();
        loadModuleFromResources("mtxOperations", "kernels/matrices.fatbin");
        loadFunction("relu", "mtxOperations");
        loadFunction("sigmoid", "mtxOperations");
        loadFunction("tangent", "mtxOperations");
        loadFunction("softmax", "mtxOperations");

        loadFunction("reluDer", "mtxOperations");
        loadFunction("sigmoidDer", "mtxOperations");
        loadFunction("tangentDer", "mtxOperations");
        loadFunction("softmaxDer", "mtxOperations");

        loadFunction("transpose", "mtxOperations");

        loadFunction("tensorAdd", "mtxOperations");
        loadFunction("tensorSub", "mtxOperations");
        loadFunction("tensorMul", "mtxOperations");
        loadFunction("tensorDiv", "mtxOperations");
        loadFunction("matrixMul", "mtxOperations");

        loadFunction("concatenateVertical", "mtxOperations");
        loadFunction("concatenateHorizontal", "mtxOperations");

        loadFunction("matrixConvEmptyBorder", "mtxOperations");
        loadFunction("matrixConvExtendBorder", "mtxOperations");
        loadFunction("matrixConvRepeatBorder", "mtxOperations");

        loadFunction("maxPooling", "mtxOperations");
        loadFunction("minPooling", "mtxOperations");
        loadFunction("avgPooling", "mtxOperations");

        addInstruction(new Add(this));
        addInstruction(new Concatenate(this));
        addInstruction(new Conv(this));
        addInstruction(new Div(this));
        addInstruction(new Linearize(this));
        addInstruction(new MatMul(this));
        addInstruction(new Mul(this));
        addInstruction(new Pooling(this));
        addInstruction(new Relu(this));
        addInstruction(new ReluDer(this));
        addInstruction(new Sigmoid(this));
        addInstruction(new SigmoidDer(this));
        addInstruction(new Softmax(this));
        addInstruction(new SoftmaxDer(this));
        addInstruction(new Sub(this));
        addInstruction(new Tangent(this));
        addInstruction(new TangentDer(this));
        addInstruction(new Transpose(this));
    }


}
