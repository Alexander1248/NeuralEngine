package ru.alexander.neuralengine;

import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.instructions.*;

import java.io.IOException;

public class MatrixEngine extends GpuExecutor {

    public MatrixEngine(boolean doublePrecision) throws IOException {
        super(doublePrecision);
        if (doublePrecision)
            loadModuleFromResources("mtxOperations", "kernels/matricesDouble.fatbin");
        else
            loadModuleFromResources("mtxOperations", "kernels/matricesFloat.fatbin");

        loadFunction("transpose", "mtxOperations");
        loadFunction("flipX", "mtxOperations");
        loadFunction("flipY", "mtxOperations");
        loadFunction("rotate90", "mtxOperations");
        loadFunction("rotate180", "mtxOperations");
        loadFunction("rotate270", "mtxOperations");
        loadFunction("set", "mtxOperations");
        loadFunction("sum", "mtxOperations");
        loadFunction("mul", "mtxOperations");

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

        addInstruction(new Add(this));
        addInstruction(new Concatenate(this));
        addInstruction(new Conv(this));
        addInstruction(new Div(this));
        addInstruction(new Linearize(this));
        addInstruction(new MatMul(this));
        addInstruction(new Mul(this));
        addInstruction(new Sub(this));
        addInstruction(new Transpose(this));
        addInstruction(new Set(this));
        addInstruction(new Transform(this));
        addInstruction(new Rotate(this));
        addInstruction(new Flip(this));
        addInstruction(new Sum(this));
    }


}
