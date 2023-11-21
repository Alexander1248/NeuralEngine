package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class PoolingBackprop extends Instruction {
    public PoolingBackprop(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "pooling_backprop";
    }

    @Override
    public void compute(String... args) {
        Matrix prevError = getVariable(args[0]);
        Matrix currError = getVariable(args[1]);
        Matrix in = getVariable(args[2]);
        Matrix out = getVariable(args[3]);

        switch (args[5]) {
            case "max", "min" -> startGPUTask("neuralOperations.maxminPoolingBackpropagation",
                    out.width(), out.height(), 1,
                    Pointer.to(new int[] { out.width() }),
                    Pointer.to(new int[] { out.height() }),
                    Pointer.to(new int[] { Integer.parseInt(args[4]) }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer()),
                    Pointer.to(currError.pointer()),
                    Pointer.to(prevError.pointer())
            );
            case "avg" -> startGPUTask("neuralOperations.avgPoolingBackpropagation",
                    out.width(), out.height(), 1,
                    Pointer.to(new int[] { out.width() }),
                    Pointer.to(new int[] { out.height() }),
                    Pointer.to(new int[] { Integer.parseInt(args[4]) }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer()),
                    Pointer.to(currError.pointer()),
                    Pointer.to(prevError.pointer())
            );
        }
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 6)
            throw new IllegalStateException("Instruction format error!");

        if (!args[5].equals("max") && !args[5].equals("min") && !args[5].equals("avg"))
            throw new IllegalStateException("Wrong border type!");

        Matrix var1 = getVariable(args[1]);
        Matrix var2 = getVariable(args[2]);
        Matrix var3 = getVariable(args[3]);

        if (var1 == null || var2 == null || var3 == null)
            throw new IllegalStateException("Variable not exists!");

        try {
            Integer.parseInt(args[4]);
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], var2.width(), var2.height()))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], var2.width(), var2.height());
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0 };
    }

    @Override
    public String documentation() {
        return """
                pooling_backprop <prev error> <curr error> <in> <out> <pooling rate> <pooling type>""";
    }
}
