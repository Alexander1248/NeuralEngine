package ru.alexander.neuralengine.instructions;

import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;

public abstract class AbstractFunctionInstruction extends Instruction {
    private final GpuExecutor executor;

    public AbstractFunctionInstruction(GpuExecutor root, GpuExecutor functionExecutor) {
        super(root);
        this.executor = functionExecutor;
    }

    @Override
    public void compute(String... args) {
        loadInstructions(args);
        executor.compute();
        unloadInstructions(args);
    }

    protected GpuExecutor getFunctionExecutor() {
        return executor;
    }

    public abstract void loadInstructions(String... args);
    public abstract void unloadInstructions(String... args);
}
