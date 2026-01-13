/**
 * ONNX Runtime Web Inference Engine for Discrete Diffusion
 *
 * This module handles in-browser text generation using the discrete diffusion model.
 * It implements the complete denoising loop from random noise to coherent text.
 */

/**
 * Performance profiler for tracking inference bottlenecks
 */
class InferenceProfiler {
    constructor() {
        this.reset();
    }

    reset() {
        this.timings = {
            modelLoad: 0,
            inference: [],
            scoreComputation: [],
            staggeredScore: [],
            transition: [],
            sampling: [],
            totalStep: []
        };
        this.generationStart = 0;
        this.generationEnd = 0;
    }

    startGeneration() {
        this.generationStart = performance.now();
    }

    endGeneration() {
        this.generationEnd = performance.now();
    }

    recordTiming(category, duration) {
        if (Array.isArray(this.timings[category])) {
            this.timings[category].push(duration);
        } else {
            this.timings[category] = duration;
        }
    }

    getStats(timingArray) {
        if (timingArray.length === 0) return { avg: 0, min: 0, max: 0, total: 0 };

        const total = timingArray.reduce((a, b) => a + b, 0);
        const avg = total / timingArray.length;
        const min = Math.min(...timingArray);
        const max = Math.max(...timingArray);

        return { avg, min, max, total };
    }

    generateReport() {
        const report = {
            totalTime: this.generationEnd - this.generationStart,
            modelLoad: this.timings.modelLoad,
            inference: this.getStats(this.timings.inference),
            scoreComputation: this.getStats(this.timings.scoreComputation),
            staggeredScore: this.getStats(this.timings.staggeredScore),
            transition: this.getStats(this.timings.transition),
            sampling: this.getStats(this.timings.sampling),
            totalStep: this.getStats(this.timings.totalStep)
        };

        // Calculate percentage breakdown
        const totalStepTime = report.totalStep.total;
        report.percentages = {
            inference: (report.inference.total / totalStepTime * 100).toFixed(1),
            scoreComputation: (report.scoreComputation.total / totalStepTime * 100).toFixed(1),
            staggeredScore: (report.staggeredScore.total / totalStepTime * 100).toFixed(1),
            transition: (report.transition.total / totalStepTime * 100).toFixed(1),
            sampling: (report.sampling.total / totalStepTime * 100).toFixed(1)
        };

        return report;
    }

    printReport() {
        const report = this.generateReport();

        console.log('\n' + '='.repeat(70));
        console.log('PERFORMANCE PROFILING REPORT');
        console.log('='.repeat(70));

        console.log(`\nTotal Generation Time: ${report.totalTime.toFixed(2)}ms`);
        console.log(`Model Load Time: ${report.modelLoad.toFixed(2)}ms`);

        console.log('\n' + '-'.repeat(70));
        console.log('PER-STEP TIMING BREAKDOWN:');
        console.log('-'.repeat(70));

        const printStats = (name, stats, percentage) => {
            console.log(`\n${name}:`);
            console.log(`  Avg: ${stats.avg.toFixed(2)}ms | Min: ${stats.min.toFixed(2)}ms | Max: ${stats.max.toFixed(2)}ms`);
            console.log(`  Total: ${stats.total.toFixed(2)}ms (${percentage}% of step time)`);
        };

        printStats('Model Inference', report.inference, report.percentages.inference);
        printStats('Score Computation', report.scoreComputation, report.percentages.scoreComputation);
        printStats('Staggered Score', report.staggeredScore, report.percentages.staggeredScore);
        printStats('Transition Kernel', report.transition, report.percentages.transition);
        printStats('Sampling', report.sampling, report.percentages.sampling);
        printStats('Total Step Time', report.totalStep, '100.0');

        console.log('\n' + '='.repeat(70));
        console.log('BOTTLENECK ANALYSIS:');
        console.log('='.repeat(70));

        // Identify bottlenecks
        const operations = [
            { name: 'Model Inference', pct: parseFloat(report.percentages.inference) },
            { name: 'Score Computation', pct: parseFloat(report.percentages.scoreComputation) },
            { name: 'Staggered Score', pct: parseFloat(report.percentages.staggeredScore) },
            { name: 'Transition Kernel', pct: parseFloat(report.percentages.transition) },
            { name: 'Sampling', pct: parseFloat(report.percentages.sampling) }
        ];

        operations.sort((a, b) => b.pct - a.pct);

        console.log('\nTime Distribution (highest to lowest):');
        operations.forEach((op, i) => {
            const bar = '█'.repeat(Math.round(op.pct / 2));
            console.log(`${i + 1}. ${op.name.padEnd(25)} ${bar} ${op.pct}%`);
        });

        console.log('\n' + '='.repeat(70));

        return report;
    }
}

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

        // Performance profiling
        this.profiler = new InferenceProfiler();
        this.enableProfiling = true; // Set to false to disable profiling overhead
    }

    /**
     * Initialize ONNX Runtime session
     */
    async init() {
        console.log('Initializing ONNX Runtime Web session...');

        const startTime = performance.now();

        // Enable WebGPU if available
        if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
            console.log('WebGPU API detected in browser');
            ort.env.wasm.proxy = false; // Disable proxy mode for WebGPU
        } else {
            console.warn('WebGPU API not available in this browser');
        }

        // Create session with WebGPU (GPU) or fallback to WebAssembly (CPU)
        // WebGPU can be 5-10x faster than WASM
        console.log('Attempting to create ONNX session...');
        console.log('ONNX Runtime version:', ort.env.versions);

        try {
            this.session = await ort.InferenceSession.create(this.modelBytes, {
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'all'
            });
            console.log('✓✓✓ SUCCESS: Using WebGPU (GPU acceleration) ✓✓✓');
            // WebGPU is active - session created successfully
            this.usingWebGPU = true;
        } catch (e) {
            console.error('WebGPU failed:', e.message);
            console.warn('Falling back to WASM (CPU)...');
            this.usingWebGPU = false;

            try {
                this.session = await ort.InferenceSession.create(this.modelBytes, {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all'
                });
                console.log('✓ Using WASM (CPU)');
            } catch (e2) {
                console.error('WASM also failed:', e2.message);
                throw new Error('Failed to create ONNX session with any provider');
            }
        }

        const loadTime = performance.now() - startTime;
        this.profiler.recordTiming('modelLoad', loadTime);

        console.log('ONNX session initialized');
        console.log(`Model load time: ${loadTime.toFixed(2)}ms`);
        console.log('Input names:', this.session.inputNames);
        console.log('Output names:', this.session.outputNames);

        if (this.usingWebGPU) {
            console.log('═══════════════════════════════════════════════════════════');
            console.log('WebGPU is ACTIVE for this session');
            console.log('Note: Not all ONNX ops have WebGPU implementations.');
            console.log('Some operations may still run on CPU.');
            console.log('═══════════════════════════════════════════════════════════');
        }
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
        const startTime = this.enableProfiling ? performance.now() : 0;

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

        if (this.enableProfiling) {
            const inferenceTime = performance.now() - startTime;
            this.profiler.recordTiming('inference', inferenceTime);
        }

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

        // Reset profiler for new generation
        if (this.enableProfiling) {
            this.profiler.reset();
            this.profiler.startGeneration();
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
            const stepStart = this.enableProfiling ? performance.now() : 0;

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

            // Run model to get log scores (already profiled internally)
            const logScore = await this.runModel(x, currSigmaBar);

            // Exponentiate to get scores
            let scoreStart = this.enableProfiling ? performance.now() : 0;
            const score = logScore.map(tokenScores =>
                tokenScores.map(s => Math.exp(s))
            );
            if (this.enableProfiling) {
                this.profiler.recordTiming('scoreComputation', performance.now() - scoreStart);
            }

            // Apply staggered score
            let stagStart = this.enableProfiling ? performance.now() : 0;
            const stagScore = this.staggeredScore(score, deltaSigma);
            if (this.enableProfiling) {
                this.profiler.recordTiming('staggeredScore', performance.now() - stagStart);
            }

            // Get transition probabilities
            let transStart = this.enableProfiling ? performance.now() : 0;
            const trans = this.transition(x, deltaSigma);
            if (this.enableProfiling) {
                this.profiler.recordTiming('transition', performance.now() - transStart);
            }

            // Multiply staggered score by transition
            const probs = stagScore.map((tokenScores, tokenIdx) =>
                tokenScores.map((s, vocabIdx) => s * trans[tokenIdx][vocabIdx])
            );

            // Sample next state
            let sampleStart = this.enableProfiling ? performance.now() : 0;
            x = this.sampleCategorical(probs);
            if (this.enableProfiling) {
                this.profiler.recordTiming('sampling', performance.now() - sampleStart);
            }

            // Total step time
            if (this.enableProfiling) {
                this.profiler.recordTiming('totalStep', performance.now() - stepStart);
            }
        }

        const finalText = this.decode(x);

        console.log('Generation complete!');
        console.log('Final text:', finalText);

        // Print profiling report
        if (this.enableProfiling) {
            this.profiler.endGeneration();
            this.profiler.printReport();
        }

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

        // Reset profiler for new generation
        if (this.enableProfiling) {
            this.profiler.reset();
            this.profiler.startGeneration();
        }

        // Denoising loop (runs exactly 'steps' times, i from 0 to steps-1)
        for (let i = 0; i < steps; i++) {
            const stepStart = this.enableProfiling ? performance.now() : 0;

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

            // Model inference (already profiled internally)
            const logScore = await this.runModel(x, currSigmaBar);

            // Score computation (exp)
            let scoreStart = this.enableProfiling ? performance.now() : 0;
            const score = logScore.map(tokenScores =>
                tokenScores.map(s => Math.exp(s))
            );
            if (this.enableProfiling) {
                this.profiler.recordTiming('scoreComputation', performance.now() - scoreStart);
            }

            // Staggered score
            let stagStart = this.enableProfiling ? performance.now() : 0;
            const stagScore = this.staggeredScore(score, deltaSigma);
            if (this.enableProfiling) {
                this.profiler.recordTiming('staggeredScore', performance.now() - stagStart);
            }

            // Transition kernel
            let transStart = this.enableProfiling ? performance.now() : 0;
            const trans = this.transition(x, deltaSigma);
            if (this.enableProfiling) {
                this.profiler.recordTiming('transition', performance.now() - transStart);
            }

            // Multiply staggered score by transition
            const probs = stagScore.map((tokenScores, tokenIdx) =>
                tokenScores.map((s, vocabIdx) => s * trans[tokenIdx][vocabIdx])
            );

            // Sampling
            let sampleStart = this.enableProfiling ? performance.now() : 0;
            x = this.sampleCategorical(probs);
            if (this.enableProfiling) {
                this.profiler.recordTiming('sampling', performance.now() - sampleStart);
            }

            // Total step time
            if (this.enableProfiling) {
                this.profiler.recordTiming('totalStep', performance.now() - stepStart);
            }

            // Yield state AFTER denoising
            yield {
                step: i + 1,
                totalSteps: steps,
                text: this.decode(x),
                sigma: currSigmaBar,
                progress: (i + 1) / (steps + 1)
            };
        }

        // Print profiling report at the end
        if (this.enableProfiling) {
            this.profiler.endGeneration();
            this.profiler.printReport();
        }
    }
}

// Export for use in HTML
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DiffusionInferenceEngine;
}
