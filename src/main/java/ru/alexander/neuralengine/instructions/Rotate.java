package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class Rotate extends Instruction {
    public Rotate(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "rotate";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);

        int angle = Math.round((float) Integer.parseInt(args[2]) / 90) % 4;
        if (angle < 0) angle += 4;

        switch (angle) {
            case 1 -> startGPUTask("mtxOperations.rotate90",
                    in.height(), in.width(), 1,
                    Pointer.to(new int[] { in.height() }),
                    Pointer.to(new int[] { in.width() }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer())
            );
            case 2 -> startGPUTask("mtxOperations.rotate180",
                    in.height(), in.width(), 1,
                    Pointer.to(new int[] { in.height() }),
                    Pointer.to(new int[] { in.width() }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer())
            );
            case 3 -> startGPUTask("mtxOperations.rotate270",
                    in.height(), in.width(), 1,
                    Pointer.to(new int[] { in.height() }),
                    Pointer.to(new int[] { in.width() }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer())
            );
        }
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 3)
            throw new IllegalStateException("Instruction format error!");

        Matrix var = getVariable(args[1]);
        if (var == null)
            throw new IllegalStateException("Variable not exists!");

        try {
            Integer.parseInt(args[2]);
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], var.height(), var.width()))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], var.height(), var.width());
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0 };
    }

    @Override
    public String documentation() {
        return """
                rotate <out> <in> <direction>""";
    }
}
