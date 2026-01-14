/**
 * ONNX Runtime Web Inference Engine for Discrete Diffusion
 *
 * This module handles in-browser text generation using the discrete diffusion model.
 * It implements the complete denoising loop from random noise to coherent text.
 */

class DiffusionInferenceEngine {
    constructor(modelBytes, vocab, config) {
        this.modelBytes = modelBytes;
        this.vocab = vocab;
        this.config = config;
        this.session = null;
        this.itos = vocab.itos;
        this.stoi = vocab.stoi;
        this.vocabSize = vocab.vocab_size;

        // Noise schedule parameters (from config.yaml)
        this.sigmaMin = 1e-4;
        this.sigmaMax = 20.0;
    }

    /**
     * Initialize ONNX Runtime session
     */
    async init() {
        // Configure WASM backend with SIMD
        ort.env.wasm.simd = true;

        // Multi-threading requires crossOriginIsolated (special HTTP headers)
        if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
            ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        } else {
            ort.env.wasm.numThreads = 1;
        }

        // Enable WebGPU if available
        if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
            ort.env.wasm.proxy = false;
        }

        // Create session with WebGPU (GPU) or fallback to WebAssembly (CPU)
        try {
            this.session = await ort.InferenceSession.create(this.modelBytes, {
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'all'
            });
            this.usingWebGPU = true;
        } catch (e) {
            this.usingWebGPU = false;
            try {
                this.session = await ort.InferenceSession.create(this.modelBytes, {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all'
                });
            } catch (e2) {
                throw new Error('Failed to create ONNX session with any provider');
            }
        }
    }

    /**
     * Geometric noise schedule
     * Implements: sigma(t) = sigma_min^(1-t) * sigma_max^t
     */
    geometricNoise(t) {
        const logSigmaMin = Math.log(this.sigmaMin);
        const logSigmaMax = Math.log(this.sigmaMax);
        const logSigma = (1 - t) * logSigmaMin + t * logSigmaMax;
        const sigma = Math.exp(logSigma);
        return { sigma, sigmaBar: sigma };
    }

    /**
     * Run model inference
     */
    async runModel(inputIds, sigma) {
        const inputIdsTensor = new ort.Tensor(
            'int64',
            BigInt64Array.from(inputIds, x => BigInt(x)),
            [1, inputIds.length]
        );

        const sigmaTensor = new ort.Tensor(
            'float32',
            new Float32Array([sigma]),
            [1, 1]
        );

        const feeds = {
            'input_ids': inputIdsTensor,
            'sigma': sigmaTensor
        };

        const results = await this.session.run(feeds);
        const logits = results.logits.data;

        // Reshape to [seq_length, vocab_size]
        const seqLength = inputIds.length;
        const logitsReshaped = [];
        for (let i = 0; i < seqLength; i++) {
            const tokenLogits = [];
            for (let j = 0; j < this.vocabSize; j++) {
                tokenLogits.push(logits[i * this.vocabSize + j]);
            }
            logitsReshaped.push(tokenLogits);
        }

        return logitsReshaped;
    }

    /**
     * Staggered score computation
     * From utils.py: staggered_score function
     */
    staggeredScore(score, deltaSigma) {
        const vocabSize = this.vocabSize;
        const expFactor = Math.exp(-deltaSigma);

        return score.map(tokenScores => {
            const scoreSum = tokenScores.reduce((a, b) => a + b, 0);
            const correction = ((expFactor - 1) / (vocabSize * expFactor)) * scoreSum;
            return tokenScores.map(s => correction + s / expFactor);
        });
    }

    /**
     * Transition kernel: probability of staying at current token vs jumping
     * From utils.py: transition function
     */
    transition(inputIds, deltaSigma) {
        const vocabSize = this.vocabSize;
        const baseProb = (1 - Math.exp(-deltaSigma)) / vocabSize;

        const transitionMatrix = [];
        for (let i = 0; i < inputIds.length; i++) {
            const currentToken = inputIds[i];
            const row = new Array(vocabSize).fill(baseProb);
            row[currentToken] = 0;
            const offDiagSum = row.reduce((a, b) => a + b, 0);
            row[currentToken] = 1 - offDiagSum;
            transitionMatrix.push(row);
        }

        return transitionMatrix;
    }

    /**
     * Sample from categorical distribution
     */
    sampleCategorical(probs) {
        const samples = [];

        for (let i = 0; i < probs.length; i++) {
            const tokenProbs = probs[i];
            const sum = tokenProbs.reduce((a, b) => a + b, 0);
            const normalized = tokenProbs.map(p => p / sum);

            const rand = Math.random();
            let cumsum = 0;
            let sampledToken = 0;

            for (let j = 0; j < normalized.length; j++) {
                cumsum += normalized[j];
                if (rand < cumsum) {
                    sampledToken = j;
                    break;
                }
            }

            samples.push(sampledToken);
        }

        return samples;
    }

    /**
     * Decode token IDs to text
     */
    decode(tokenIds) {
        return tokenIds.map(id => this.itos[id]).join('');
    }

    /**
     * Encode text to token IDs
     */
    encode(text) {
        return text.split('').map(char => this.stoi[char] || 0);
    }

    /**
     * Generate a confession with intermediate diffusion steps
     */
    async generate({
        contextLength = 256,
        steps = 128,
        captureSteps = null,
        onProgress = null
    } = {}) {
        if (!captureSteps) {
            captureSteps = [];
            for (let i = 0; i <= steps; i += 16) {
                captureSteps.push(i);
            }
            captureSteps.push(steps);
        }

        const eps = 1e-5;
        const stepSize = (1 - eps) / steps;

        let x = new Array(contextLength).fill(0).map(() =>
            Math.floor(Math.random() * this.vocabSize)
        );

        const capturedFrames = {};

        for (let i = 0; i <= steps; i++) {
            const t = 1 - i * stepSize;
            const { sigmaBar: currSigmaBar } = this.geometricNoise(t);

            if (captureSteps.includes(i)) {
                capturedFrames[i] = this.decode(x);
            }

            if (onProgress) {
                onProgress(i, steps, this.decode(x));
            }

            let deltaSigma;
            if (i < steps) {
                const { sigmaBar: nextSigmaBar } = this.geometricNoise(t - stepSize);
                deltaSigma = currSigmaBar - nextSigmaBar;
            } else {
                deltaSigma = currSigmaBar;
            }

            const logScore = await this.runModel(x, currSigmaBar);
            const score = logScore.map(tokenScores =>
                tokenScores.map(s => Math.exp(s))
            );

            const stagScore = this.staggeredScore(score, deltaSigma);
            const trans = this.transition(x, deltaSigma);

            const probs = stagScore.map((tokenScores, tokenIdx) =>
                tokenScores.map((s, vocabIdx) => s * trans[tokenIdx][vocabIdx])
            );

            x = this.sampleCategorical(probs);
        }

        const finalText = this.decode(x);

        return {
            text: finalText,
            steps: capturedFrames,
            allSteps: captureSteps
        };
    }

    /**
     * Generate with real-time streaming (for animation)
     * Yields each denoising step
     */
    async* generateStream(contextLength = 256, steps = 128) {
        const eps = 1e-5;
        const stepSize = (1 - eps) / steps;

        let x = new Array(contextLength).fill(0).map(() =>
            Math.floor(Math.random() * this.vocabSize)
        );

        for (let i = 0; i < steps; i++) {
            const t = 1 - (i + 1) * stepSize;
            const { sigmaBar: currSigmaBar } = this.geometricNoise(t);

            let deltaSigma;
            if (i < steps - 1) {
                const { sigmaBar: nextSigmaBar } = this.geometricNoise(t - stepSize);
                deltaSigma = currSigmaBar - nextSigmaBar;
            } else {
                deltaSigma = currSigmaBar;
            }

            const logScore = await this.runModel(x, currSigmaBar);
            const score = logScore.map(tokenScores =>
                tokenScores.map(s => Math.exp(s))
            );

            const stagScore = this.staggeredScore(score, deltaSigma);
            const trans = this.transition(x, deltaSigma);

            const probs = stagScore.map((tokenScores, tokenIdx) =>
                tokenScores.map((s, vocabIdx) => s * trans[tokenIdx][vocabIdx])
            );

            x = this.sampleCategorical(probs);

            yield {
                step: i + 1,
                totalSteps: steps,
                text: this.decode(x),
                sigma: currSigmaBar,
                progress: (i + 1) / (steps + 1)
            };
        }
    }
}

// Export for use in HTML
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DiffusionInferenceEngine;
}
