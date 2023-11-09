package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class Div extends Instruction {
    public Div(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "div";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in1 = getVariable(args[1]);
        Matrix in2 = getVariable(args[2]);

        startGPUTask("mtxOperations.tensorDiv",
                in1.width(), in1.height(), 1,
                Pointer.to(new int[] { in1.width() }),
                Pointer.to(new int[] { in1.height() }),
                Pointer.to(in1.pointer()),
                Pointer.to(in2.pointer()),
                Pointer.to(out.pointer())
                );
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 3)
            throw new IllegalStateException("Instruction format error!");

        Matrix var1 = getVariable(args[1]);
        Matrix var2 = getVariable(args[2]);

        if (var1 == null || var2 == null)
            throw new IllegalStateException("Variable not exists!");

        if (var1.width() != var2.width())
            throw new IllegalStateException("Widths not equal!");

        if (var1.height() != var2.height())
            throw new IllegalStateException("Heights not equal!");

        addVariable(args[0], var1.width(), var1.height());
    }
}
