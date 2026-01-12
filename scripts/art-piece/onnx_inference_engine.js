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
        console.log('Initializing ONNX Runtime Web session...');

        // Create session with WebAssembly execution provider
        this.session = await ort.InferenceSession.create(this.modelBytes, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        console.log('ONNX session initialized');
        console.log('Input names:', this.session.inputNames);
        console.log('Output names:', this.session.outputNames);
    }

    /**
     * Geometric noise schedule
     * Implements: sigma(t) = sigma_min^(1-t) * sigma_max^t
     */
    geometricNoise(t) {
        // t is a scalar between 0 and 1
        const logSigmaMin = Math.log(this.sigmaMin);
        const logSigmaMax = Math.log(this.sigmaMax);
        const logSigma = (1 - t) * logSigmaMin + t * logSigmaMax;
        const sigma = Math.exp(logSigma);

        // Return sigma and sigma_bar (simplified, assuming identity transform)
        return { sigma, sigmaBar: sigma };
    }

    /**
     * Run model inference
     */
    async runModel(inputIds, sigma) {
        // Create input tensors
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

        // Run inference
        const feeds = {
            'input_ids': inputIdsTensor,
            'sigma': sigmaTensor
        };

        const results = await this.session.run(feeds);

        // Get logits output
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
     *
     * Applies: exp(-delta_sigma * Q^tok) * score
     * = correction + score / exp_factor
     * where correction = ((exp_factor - 1) / (vocab_size * exp_factor)) * sum(score)
     */
    staggeredScore(score, deltaSigma) {
        const vocabSize = this.vocabSize;
        const expFactor = Math.exp(-deltaSigma);

        return score.map(tokenScores => {
            // Sum across vocabulary for this token position
            const scoreSum = tokenScores.reduce((a, b) => a + b, 0);

            // Compute correction term
            const correction = ((expFactor - 1) / (vocabSize * expFactor)) * scoreSum;

            // Apply staggered score formula
            return tokenScores.map(s => correction + s / expFactor);
        });
    }

    /**
     * Transition kernel: probability of staying at current token vs jumping
     * From utils.py: transition function
     *
     * Computes: exp(delta_sigma * Q^tok)(x_t, y)
     * Uniform mixing with base_prob = (1 - exp(-delta_sigma)) / vocab_size for off-diagonal
     * Diagonal is set to ensure probabilities sum to 1
     */
    transition(inputIds, deltaSigma) {
        const vocabSize = this.vocabSize;
        const baseProb = (1 - Math.exp(-deltaSigma)) / vocabSize;

        const transitionMatrix = [];
        for (let i = 0; i < inputIds.length; i++) {
            const currentToken = inputIds[i];
            const row = new Array(vocabSize).fill(baseProb);

            // Set diagonal to 0 initially
            row[currentToken] = 0;

            // Sum of off-diagonal elements
            const offDiagSum = row.reduce((a, b) => a + b, 0);

            // Set diagonal to ensure row sums to 1
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

            // Normalize
            const sum = tokenProbs.reduce((a, b) => a + b, 0);
            const normalized = tokenProbs.map(p => p / sum);

            // Sample
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
     *
     * @param {number} contextLength - Length of sequence to generate
     * @param {number} steps - Number of denoising steps
     * @param {Array} captureSteps - Which steps to capture for animation
     * @param {Function} onProgress - Callback for progress updates
     * @returns {Object} Generated text and intermediate steps
     */
    async generate({
        contextLength = 256,
        steps = 128,
        captureSteps = null,
        onProgress = null
    } = {}) {
        // If captureSteps not specified, capture every 16 steps
        if (!captureSteps) {
            captureSteps = [];
            for (let i = 0; i <= steps; i += 16) {
                captureSteps.push(i);
            }
            captureSteps.push(steps); // Always include final step
        }

        const eps = 1e-5;
        const stepSize = (1 - eps) / steps;

        // Initialize with random tokens
        let x = new Array(contextLength).fill(0).map(() =>
            Math.floor(Math.random() * this.vocabSize)
        );

        const capturedFrames = {};

        console.log('Starting diffusion generation...');
        console.log(`Steps: ${steps}, Context length: ${contextLength}`);

        // Denoising loop
        for (let i = 0; i <= steps; i++) {
            const t = 1 - i * stepSize;
            const { sigmaBar: currSigmaBar } = this.geometricNoise(t);

            // Capture frame if needed
            if (captureSteps.includes(i)) {
                capturedFrames[i] = this.decode(x);
                console.log(`Step ${i}/${steps}: ${capturedFrames[i].substring(0, 50)}...`);
            }

            // Progress callback
            if (onProgress) {
                onProgress(i, steps, this.decode(x));
            }

            // Denoising step
            let deltaSigma;
            if (i < steps) {
                const { sigmaBar: nextSigmaBar } = this.geometricNoise(t - stepSize);
                deltaSigma = currSigmaBar - nextSigmaBar;
            } else {
                // Last denoising step - use full current sigma
                deltaSigma = currSigmaBar;
            }

            // Run model to get log scores
            const logScore = await this.runModel(x, currSigmaBar);

            // Exponentiate to get scores
            const score = logScore.map(tokenScores =>
                tokenScores.map(s => Math.exp(s))
            );

            // Apply staggered score
            const stagScore = this.staggeredScore(score, deltaSigma);

            // Get transition probabilities
            const trans = this.transition(x, deltaSigma);

            // Multiply staggered score by transition
            const probs = stagScore.map((tokenScores, tokenIdx) =>
                tokenScores.map((s, vocabIdx) => s * trans[tokenIdx][vocabIdx])
            );

            // Sample next state
            x = this.sampleCategorical(probs);
        }

        const finalText = this.decode(x);

        console.log('Generation complete!');
        console.log('Final text:', finalText);

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

        // Denoising loop (runs exactly 'steps' times, i from 0 to steps-1)
        for (let i = 0; i < steps; i++) {
            const t = 1 - (i + 1) * stepSize;
            const { sigmaBar: currSigmaBar } = this.geometricNoise(t);

            // Denoising step
            let deltaSigma;
            if (i < steps - 1) {
                const { sigmaBar: nextSigmaBar } = this.geometricNoise(t - stepSize);
                deltaSigma = currSigmaBar - nextSigmaBar;
            } else {
                // Last denoising step - use full current sigma
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

            // Yield state AFTER denoising
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
