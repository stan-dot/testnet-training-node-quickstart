import React, { useState } from 'react';
import axios from 'axios';

const LoraTrainingForm = () => {
  const [modelName, setModelName] = useState("model1");
  const [perDeviceTrainBatchSize, setPerDeviceTrainBatchSize] = useState(1);
  const [gradientAccumulationSteps, setGradientAccumulationSteps] = useState(8);
  const [numTrainEpochs, setNumTrainEpochs] = useState(1);
  const [loraRank, setLoraRank] = useState(8);
  const [loraAlpha, setLoraAlpha] = useState(16);
  const [loraDropoutRate, setLoraDropoutRate] = useState(0.8);

  const handleSubmit = async (e: any) => {
    e.preventDefault();
    const data = {
      model_name: modelName,
      per_device_train_batch_size: perDeviceTrainBatchSize,
      gradient_accumulation_steps: gradientAccumulationSteps,
      num_train_epochs: numTrainEpochs,
      lora_rank: loraRank,
      lora_alpha: loraAlpha,
      lora_dropout_rate: loraDropoutRate
    };

    try {
      const response = await axios.post("http://localhost:8000/train-lora", data);
      console.log("Training Response:", response.data);
      alert("Training initiated: " + response.data.message);
    } catch (error) {
      console.error("Error initiating training:", error);
      alert("Failed to initiate training.");
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Model Name:
        <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
          <option value="Qwen/Qwen1.5-0.5B">"Qwen/Qwen1.5-0.5B</option>
          <option value="Qwen/Qwen1.5-1.8B">"Qwen/Qwen1.5-1.8B</option>
          <option value="Qwen/Qwen1.5-7B">"Qwen/Qwen1.5-7B</option>
          <option value="google/gemma-2b">"google/gemma-2b</option>
          <option value="google/gemma-7b">"google/gemma-7b</option>
        </select>
      </label>
      <br />

      <label>
        Per Device Train Batch Size:
        <input
          type="number"
          value={perDeviceTrainBatchSize}
          onChange={(e) => setPerDeviceTrainBatchSize(parseInt(e.target.value))}
          min="1"
          step="1"
        />
      </label>
      <br />

      <label>
        Gradient Accumulation Steps:
        <input
          type="number"
          value={gradientAccumulationSteps}
          onChange={(e) => setGradientAccumulationSteps(parseInt(e.target.value))}
          min="1"
          step="1"
        />
      </label>
      <br />

      <label>
        Number of Training Epochs:
        <input
          type="number"
          value={numTrainEpochs}
          onChange={(e) => setNumTrainEpochs(parseInt(e.target.value))}
          min="1"
          step="1"
        />
      </label>
      <br />

      <label>
        LoRA Rank:
        <input
          type="number"
          value={loraRank}
          onChange={(e) => setLoraRank(parseInt(e.target.value))}
          min="1"
          step="1"
        />
      </label>
      <br />

      <label>
        LoRA Alpha:
        <input
          type="number"
          value={loraAlpha}
          onChange={(e) => setLoraAlpha(parseInt(e.target.value))}
          min="1"
          step="1"
        />
      </label>
      <br />

      <label>
        LoRA Dropout Rate:
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={loraDropoutRate}
          onChange={(e) => setLoraDropoutRate(parseFloat(e.target.value))}
        />
        <span>{loraDropoutRate}</span>
      </label>
      <br />

      <button type="submit">Start Training</button>
    </form>
  );
};

export default LoraTrainingForm;
