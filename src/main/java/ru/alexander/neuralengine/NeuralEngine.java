package ru.alexander.neuralengine;

import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.instructions.*;

import java.io.IOException;

public class NeuralEngine extends GpuExecutor {

    //    nvcc -m64 -arch=all-major -fatbin matrices.cu -o matrices.fatbin
    public NeuralEngine() throws IOException {
        super();
        loadModuleFromResources("mtxOperations", "kernels/matrices.fatbin");
        loadFunction("relu", "mtxOperations");
        loadFunction("sigmoid", "mtxOperations");
        loadFunction("tangent", "mtxOperations");
        loadFunction("softmax", "mtxOperations");

        loadFunction("reluDet", "mtxOperations");
        loadFunction("sigmoidDet", "mtxOperations");
        loadFunction("tangentDet", "mtxOperations");
        loadFunction("softmaxDet", "mtxOperations");

        loadFunction("transpose", "mtxOperations");

        loadFunction("tensorAdd", "mtxOperations");
        loadFunction("tensorSub", "mtxOperations");
        loadFunction("concatenate", "mtxOperations");
        loadFunction("tensorMul", "mtxOperations");
        loadFunction("tensorDiv", "mtxOperations");
        loadFunction("matrixMul", "mtxOperations");
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
        addInstruction(new ReluDet(this));
        addInstruction(new Sigmoid(this));
        addInstruction(new SigmoidDet(this));
        addInstruction(new Softmax(this));
        addInstruction(new SoftmaxDet(this));
        addInstruction(new Sub(this));
        addInstruction(new Tangent(this));
        addInstruction(new TangentDet(this));
        addInstruction(new Transpose(this));
    }
}
