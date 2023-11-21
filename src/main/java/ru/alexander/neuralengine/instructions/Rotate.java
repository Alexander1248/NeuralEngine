package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import jcuda.Sizeof;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

import static jcuda.driver.JCudaDriver.cuMemcpyDtoD;

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
            case 0 -> cuMemcpyDtoD(out.pointer(), in.pointer(), (long) Sizeof.FLOAT * in.width() * in.height());
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
            default -> throw new IllegalStateException("Unexpected value: " + angle);
        }
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 3)
            throw new IllegalStateException("Instruction format error!");

        Matrix var = getVariable(args[1]);
        if (var == null)
            throw new IllegalStateException("Variable not exists!");

        int angle;
        try {
            angle = Math.round((float) Integer.parseInt(args[2]) / 90) % 4;
            if (angle < 0) angle += 4;
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }
        int w, h;
        switch (angle) {
            case 0, 2 -> {
                w = var.width();
                h = var.height();
            }
            case 1, 3 -> {
                w = var.height();
                h = var.width();
            }
            default -> throw new IllegalStateException("Unexpected value: " + angle);
        }
        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], w, h))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], w, h);
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
