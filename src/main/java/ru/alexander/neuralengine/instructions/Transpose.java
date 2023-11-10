package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class Transpose extends Instruction {
    public Transpose(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "transpose";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);

        startGPUTask("mtxOperations.transpose",
                in.height(), in.width(), 1,
                Pointer.to(new int[] { in.height() }),
                Pointer.to(new int[] { in.width() }),
                Pointer.to(in.pointer()),
                Pointer.to(out.pointer())
                );
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 2)
            throw new IllegalStateException("Instruction format error!");

        Matrix var = getVariable(args[1]);
        if (var == null)
            throw new IllegalStateException("Variable not exists!");

        addVariable(args[0], var.height(), var.width());
    }
    public String getOutputVariableArg(String... args) {
        return args[0];
    }

    @Override
    public String documentation() {
        return """
                transpose <out> <in>""";
    }
}
