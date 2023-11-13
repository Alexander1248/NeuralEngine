package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class ReluDer extends Instruction {
    public ReluDer(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "relu_der";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);

        startGPUTask("mtxOperations.reluDer",
                in.width(), in.height(), 1,
                Pointer.to(new int[] { in.width() }),
                Pointer.to(new int[] { in.height() }),
                Pointer.to(new float[] { Float.parseFloat(args[2]) }),
                Pointer.to(new float[] { Float.parseFloat(args[3]) }),
                Pointer.to(in.pointer()),
                Pointer.to(out.pointer())
                );
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 4)
            throw new IllegalStateException("Instruction format error!");

        Matrix var = getVariable(args[1]);
        if (var == null)
            throw new IllegalStateException("Variable not exists!");

        try {
            Float.parseFloat(args[2]);
            Float.parseFloat(args[3]);
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], var.width(), var.height()))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], var.width(), var.height());
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0 };
    }

    @Override
    public String documentation() {
        return """
                relu_det <out> <in> <positive coefficient> <negative coefficient>""";
    }
}
