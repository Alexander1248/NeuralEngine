package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class Flip extends Instruction {
    public Flip(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "flip";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);

        switch (args[2]) {
            case "x" -> startGPUTask("mtxOperations.flipX",
                    in.width(), in.height(), 1,
                    Pointer.to(new int[] { in.width() }),
                    Pointer.to(new int[] { in.height() }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer())
            );
            case "y" -> startGPUTask("mtxOperations.flipY",
                    in.width(), in.height(), 1,
                    Pointer.to(new int[] { in.width() }),
                    Pointer.to(new int[] { in.height() }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer())
            );
        }
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 3)
            throw new IllegalStateException("Instruction format error!");

        if (!args[2].equals("x") && !args[2].equals("y"))
            throw new IllegalStateException("Wrong border type!");

        Matrix var = getVariable(args[1]);
        if (var == null)
            throw new IllegalStateException("Variable not exists!");

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
                flip <out> <in> <direction>""";
    }
}
