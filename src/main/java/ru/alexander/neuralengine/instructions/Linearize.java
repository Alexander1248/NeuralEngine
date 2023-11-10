package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import jcuda.Sizeof;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

import static jcuda.driver.JCudaDriver.*;

public class Linearize extends Instruction {
    public Linearize(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "linearize";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);
        cuMemcpyDtoD(out.pointer(), in.pointer(), (long) Sizeof.FLOAT * in.width());
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 2)
            throw new IllegalStateException("Instruction format error!");

        Matrix var = getVariable(args[1]);
        if (var == null)
            throw new IllegalStateException("Variable not exists!");

        addVariable(args[0], var.width() * var.height(), 1);
    }

    @Override
    public String documentation() {
        return """
                linearize <out> <in>""";
    }
}
