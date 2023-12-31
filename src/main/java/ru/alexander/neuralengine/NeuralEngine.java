package ru.alexander.neuralengine;

import ru.alexander.neuralengine.instructions.*;

import java.io.IOException;

public class NeuralEngine extends MatrixEngine {

    public NeuralEngine(boolean doublePrecision) throws IOException {
        super(doublePrecision);
        if (doublePrecision)
            loadModuleFromResources("neuralOperations", "kernels/neuralDouble.fatbin");
        else
            loadModuleFromResources("neuralOperations", "kernels/neuralFloat.fatbin");

        loadFunction("relu", "neuralOperations");
        loadFunction("sigmoid", "neuralOperations");
        loadFunction("tangent", "neuralOperations");
        loadFunction("softmax", "neuralOperations");

        loadFunction("reluDer", "neuralOperations");
        loadFunction("sigmoidDer", "neuralOperations");
        loadFunction("tangentDer", "neuralOperations");
        loadFunction("softmaxDer", "neuralOperations");

        loadFunction("matrixMulBackpropagationErrorTraversal", "neuralOperations");
        loadFunction("matrixMulBackpropagationWeightCorrection", "neuralOperations");

        loadFunction("matrixConvEmptyBorderBackpropagationErrorTraversal", "neuralOperations");
        loadFunction("matrixConvExtendBorderBackpropagationErrorTraversal", "neuralOperations");
        loadFunction("matrixConvRepeatBorderBackpropagationErrorTraversal", "neuralOperations");

        loadFunction("matrixConvEmptyBorderBackpropagationWeightCorrection", "neuralOperations");
        loadFunction("matrixConvExtendBorderBackpropagationWeightCorrection", "neuralOperations");
        loadFunction("matrixConvRepeatBorderBackpropagationWeightCorrection", "neuralOperations");

        loadFunction("maxPooling", "neuralOperations");
        loadFunction("minPooling", "neuralOperations");
        loadFunction("avgPooling", "neuralOperations");
        loadFunction("maxminPoolingBackpropagation", "neuralOperations");
        loadFunction("avgPoolingBackpropagation", "neuralOperations");


        addInstruction(new ConvBackprop(this));
        addInstruction(new MatMulBackprop(this));
        addInstruction(new Pooling(this));
        addInstruction(new PoolingBackprop(this));
        addInstruction(new Relu(this));
        addInstruction(new ReluDer(this));
        addInstruction(new Sigmoid(this));
        addInstruction(new SigmoidDer(this));
        addInstruction(new Softmax(this));
        addInstruction(new SoftmaxDer(this));
        addInstruction(new Tangent(this));
        addInstruction(new TangentDer(this));
    }


}
