package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class SigmoidDet extends Instruction {
    public SigmoidDet(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "sigmoid_det";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);

        startGPUTask("mtxOperations.sigmoidDet",
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

        addVariable(args[0], var.width(), var.height());
    }
}
