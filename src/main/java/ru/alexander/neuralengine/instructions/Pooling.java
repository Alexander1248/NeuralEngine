package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class Pooling extends Instruction {
    public Pooling(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "pooling";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);

        switch (args[3]) {
            case "max" -> startGPUTask("mtxOperations.maxPooling",
                    out.width(), out.height(), 1,
                    Pointer.to(new int[] { out.width() }),
                    Pointer.to(new int[] { out.height() }),
                    Pointer.to(new int[] { Integer.parseInt(args[2]) }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer())
            );
            case "min" -> startGPUTask("mtxOperations.minPooling",
                    out.width(), out.height(), 1,
                    Pointer.to(new int[] { out.width() }),
                    Pointer.to(new int[] { out.height() }),
                    Pointer.to(new int[] { Integer.parseInt(args[2]) }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer())
            );
            case "avg" -> startGPUTask("mtxOperations.avgPooling",
                    out.width(), out.height(), 1,
                    Pointer.to(new int[] { out.width() }),
                    Pointer.to(new int[] { out.height() }),
                    Pointer.to(new int[] { Integer.parseInt(args[2]) }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer())
            );
        }
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 4)
            throw new IllegalStateException("Instruction format error!");

        if (!args[3].equals("max") && !args[3].equals("min") && !args[3].equals("avg"))
            throw new IllegalStateException("Wrong border type!");

        Matrix var = getVariable(args[1]);

        if (var == null)
            throw new IllegalStateException("Variable not exists!");

        int rate;
        try {
            rate = Integer.parseInt(args[2]);
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }

        if (var.width() % rate != 0)
            throw new IllegalStateException("Width is not divisible!");

        if (var.height() % rate != 0)
            throw new IllegalStateException("Height is not divisible!");

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], var.width() / rate, var.height() / rate))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], var.width() / rate, var.height() / rate);
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0 };
    }

    @Override
    public String documentation() {
        return """
                pooling <out> <in> <pooling rate> <pooling type>""";
    }
}
