package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class TangentDet extends Instruction {
    public TangentDet(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "tangent_det";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);

        startGPUTask("mtxOperations.tangentDet",
                in.width(), in.height(), 1,
                Pointer.to(new int[] { in.width() }),
                Pointer.to(new int[] { in.height() }),
                Pointer.to(new float[] { Float.parseFloat(args[2]) }),
                Pointer.to(in.pointer()),
                Pointer.to(out.pointer())
                );
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 3)
            throw new IllegalStateException("Instruction format error!");

        Matrix var = getVariable(args[1]);
        if (var == null)
            throw new IllegalStateException("Variable not exists!");

        try {
            Float.parseFloat(args[2]);
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], var.width(), var.height()))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], var.width(), var.height());
    }
    public String getOutputVariableArg(String... args) {
        return args[0];
    }

    @Override
    public String documentation() {
        return """
                tangent_det <out> <in> <force>""";
    }
}
