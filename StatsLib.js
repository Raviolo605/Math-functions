


const alpha = 0.05; // Significance level for hypothesis testing
const GlobalBound = 0.5; // Threshold for correlation value
const math = require('mathjs'); // Importing math library for matrix operations
const PCA = require('ml-pca');
// Function to provide access to various statistical functions


function GiveMe(key) {
    switch (key) {
		
        // Basic statistical functions
        case 'mean': return mean;
        case 'prodDiff': return prodDiff;
        case 'squaresDiff': return squaresDiff;
        case 'sum': return sum;
        case 'variance': return variance;
        case 'standardDeviation': return standardDeviation;
		case 'weightedMean': return weightedMean;
        case 'rSquared': return rSquared;
        case 'adjustedRSquared': return adjustedRSquared;
        case 'rmse': return rmse;
        case 'mad': return mad;
		case 'geometricMean': return geometricMean;		
        // Hypothesis testing functions
        case 't': return t_studentTest;
        case 'z': return gaussianTest;
        case 'x': return chiSquareTest;
        case 't_probability': return t_studentTestProbabilistic;
        case 'gaussian_probability': return gaussianTestProbabilistic;
        case 'chiSquare_probability': return chiSquareTestProbabilistic;		
        // Correlation and covariance functions		
        case 'covariance': return covariance;
        case 'correlation': return correlation;
		// Interpolation functions	
		case 'linearInterpolation': return linearInterpolation;
		case 'polynomialInterpolation': return polynomialInterpolation;		
        // Regression functions
		case 'linearRegression': return linearRegression;
        case 'polynomialRegression': return polynomialRegression;
		case 'logisticRegression': return logisticRegression;
		case 'exponentialRegression': return exponentialRegression;
        case 'logarithmicRegression': return logarithmicRegression;
        case 'powerRegression': return powerRegression;
        // Kalman filter related functions
        case 'extendedKalmanFilter': return extendedKalmanFilter;
        // Z-transform function
        case 'Z_transform': return Z_transform;
		// Cluster Analysis
		case 'clusterAnalysis': return clusterAnalysis;
        // Time series analysis
        case 'timeSeriesAnalysis': return timeSeriesAnalysis;
        // Frequency analysis
        case 'frequencyAnalysis': return frequencyAnalysis;
        // Matrix manipulation function
        case 'matrix': return matrix;
        case 'matrixMultiply': return matrixMultiply;
        case 'matrixTranspose': return matrixTranspose;
        case 'matrixAdd': return matrixAdd;
        case 'matrixIdentity': return matrixIdentity;
        case 'matrixInverse': return matrixInverse;
		// Advanced functions
        case 'discreteFourierTransform': return discreteFourierTransform;
        case 'inverseDiscreteFourierTransform': return inverseDiscreteFourierTransform;
        case 'convolution': return convolution;
        case 'gradientDescent': return gradientDescent;
		case 'principalComponentAnalysis': return principalComponentAnalysis;
        case 'monteCarloSimulation': return monteCarloSimulation;
        default: return;
		// Data preprocessing functions
        case 'minMaxScaling': return minMaxScaling;
        case 'standardization': return standardization;
        case 'normalization': return normalization;
        case 'logTransformation': return logTransformation;
        case 'boxCoxTransformation': return boxCoxTransformation;
        case 'featureScaling': return featureScaling;
		// Distance calculation functions
        case 'euclideanDistance': return euclideanDistance;
        case 'manhattanDistance': return manhattanDistance;
        case 'chebyshevDistance': return chebyshevDistance;
		// Machine Learning functions
		case 'kNearestNeighbors': return kNearestNeighbors
		
    }
}

// Basic statistical functions
function sum(arr, index, total) {
    if (index === arr.length) return total;
    else return GiveMe('sum')(arr, index + 1, total + arr[index]);
}

function mean(arr) {
    return sum(arr, 0, 0) / arr.length;
}

function squaredDifference(arr, meanValue, index, result) {
    if (index === arr.length) return result;
    else {
        result.push(Math.pow(arr[index] - meanValue, 2));
        return GiveMe('squaresDiff')(arr, meanValue, index + 1, result);
    }
}

function variance(arr) {
    const arrMean = GiveMe('mean')(arr);
    return sum(squaredDifference(arr, arrMean, 0, [])) / arr.length;
}

function standardDeviation(arr) {
    return Math.sqrt(variance(arr));
}
// Geometric mean
function geometricMean(arr) {
    const product = arr.reduce((acc, val) => acc * val, 1);
    return Math.pow(product, 1 / arr.length);
}
// Absolute mean deviation
function mad(observed, predicted) {
    const absoluteDeviations = observed.map((val, index) => Math.abs(val - predicted[index]));
    return GiveMe('mean')(absoluteDeviations);
}
//Weighted mean function
function weightedMean(arr, weights) {
    if (arr.length !== weights.length) {
        throw new Error('Array and weights must have the same length.');
    }
    const weightedSum = arr.reduce((sum, val, index) => sum + val * weights[index], 0);
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
    return weightedSum / totalWeight;
}


// Hypothesis testing functions
function t_studentTest(sampleData, populationMean, index = 0, sum = 0) {
    if (index === sampleData.length) {
        const tStat = sum / (sampleData.length - 1) / Math.sqrt(sampleData.length);
        return tStat < -alpha || tStat > alpha;
    }

    const currentDiff = sampleData[index] - populationMean;
    return GiveMe('t')(sampleData, populationMean, index + 1, sum + Math.pow(currentDiff, 2));
}

function gaussianTest(sampleData, index = 0, sum = 0) {
    if (index === sampleData.length) {
        const zScore = sum / Math.sqrt(sampleData.length - 1);
        return zScore < -alpha || zScore > alpha;
    }

    const currentVal = sampleData[index];
    const currentDiff = Math.pow(currentVal - mean(sampleData), 2);
    return GiveMe('z')(sampleData, index + 1, sum + currentDiff);
}

function chiSquareTest(observed, expected, index = 0, sum = 0) {
    if (index === observed.length) {
        const chiSquareStat = sum;
        return chiSquareStat > alpha;
    }

    const observedVal = observed[index];
    const expectedVal = expected[index];
    const currentDiff = Math.pow(observedVal - expectedVal, 2) / expectedVal;
    return GiveMe('x')(observed, expected, index + 1, sum + currentDiff);
}

// Product of differences function
function prodDiff(v1, v2, mean1, mean2, index, result) {
    if (index === v1.length) return result;
    else {
        result.push((v1[index] - mean1) * (v2[index] - mean2));
        return GiveMe('prodDiff')(v1, v2, mean1, mean2, index + 1, result);
    }
}

// Squared differences function
function squaresDiff(array, mean, index, result) {
    if (index === array.length) return result;
    else {
        result.push(Math.pow(array[index] - mean, 2));
        return GiveMe('squaresDiff')(array, mean, index + 1, result);
    }
}

// Covariance calculation function
function covariance(v1, v2) {
    const meanV1 = GiveMe('mean')(v1);
    const meanV2 = GiveMe('mean')(v2);

    const prodDifferences = GiveMe('prodDiff')(v1, v2, meanV1, meanV2, 0, []);

    const covarianceValue = GiveMe('sum')(prodDifferences, 0, 0) / v1.length;

    return covarianceValue;
}
// Linear interpolation function
function linearInterpolation(x, y, xInterp) {
    // Calcola la regressione lineare per ottenere la pendenza e l'intercetta
    const { slope, intercept } = GiveMe('linearRegression')(x, y);

    // Calcola il valore interpolato utilizzando l'equazione della retta
    const yInterp = slope * xInterp + intercept;

    return yInterp;
}
// Polynomial interpolation function
function polynomialInterpolation(x, y, xInterp, degree) {
    // Coeffcients
    const coefficients = GiveMe('polynomialRegression')(x, y, degree);

    // Interpolation values
    const yInterp = xInterp.map((xValue) => {
        let yValue = 0;
        for (let i = 0; i <= degree; i++) {
            yValue += coefficients[i][0] * Math.pow(xValue, i);
        }
        return yValue;
    });

    return yInterp;
}
// rSquared function
function rSquared(observed, predicted) {
    const meanObserved = GiveMe('mean')(observed);
    const totalVariation = GiveMe('sum')(observed.map(val => Math.pow(val - meanObserved, 2)));
    const residualVariation = GiveMe('sum')(observed.map((val, index) => Math.pow(val - predicted[index], 2)));
    return 1 - (residualVariation / totalVariation);
}
// rSquared adjusted function
function adjustedRSquared(observed, predicted, numPredictors) {
    const n = observed.length;
    const rSquaredValue = rSquared(observed, predicted);
    return 1 - (1 - rSquaredValue) * ((n - 1) / (n - numPredictors - 1));
}

// Root mean square error
function rmse(observed, predicted) {
    const squaredErrors = observed.map((val, index) => Math.pow(val - predicted[index], 2));
    const meanSquaredError = GiveMe('mean')(squaredErrors);
    return Math.sqrt(meanSquaredError);
}
// Linear regression function
function linearRegression(v1, v2) {
    const meanV1 = GiveMe('mean')(v1);
    const meanV2 = GiveMe('mean')(v2);

    const prodDifferences = GiveMe('prodDiff')(v1, v2, meanV1, meanV2, 0, []);
    const squaresDiffV1 = GiveMe('squaresDiff')(v1, meanV1, 0, []);

    const slope = GiveMe('sum')(prodDifferences, 0, 0) / GiveMe('sum')(squaresDiffV1, 0, 0);
    const intercept = meanV2 - slope * meanV1;

    return { slope, intercept };
}
// COrrelation
function correlation(v1, v2) {
    const meanV1 = GiveMe('mean')(v1);
    const meanV2 = GiveMe('mean')(v2);

    const prodDifferences = GiveMe('prodDiff')(v1, v2, meanV1, meanV2, 0, []);
    const squaresDiffV1 = GiveMe('squaresDiff')(v1, meanV1, 0, []);
    const squaresDiffV2 = GiveMe('squaresDiff')(v2, meanV2, 0, []);

    const numerator = GiveMe('sum')(prodDifferences, 0, 0);
    const denominator = Math.sqrt(GiveMe('sum')(squaresDiffV1, 0, 0) * GiveMe('sum')(squaresDiffV2, 0, 0));

    const correlationValue = numerator / denominator;

    if (Math.abs(correlationValue) > GlobalBound) return Math.sign(correlationValue);
    else return 0;
}
//Logistic Regression
function logisticRegression(x, y) {
    // Perform linear regression to get coefficients
    const { slope, intercept } = GiveMe('linearRegression')(x, y);

    // Calculate probabilities using sigmoid function
    const probabilities = x.map((xValue) => GiveMe('sigmoid')(intercept + slope * xValue));

    return probabilities;
}
// Polynomial regression function
function polynomialRegression(x, y, degree) {
    if (x.length !== y.length || x.length === 0 || degree < 1) {
        throw new Error('Invalid input: x and y must have the same length, and degree must be at least 1');
    }

    // Costruisci la matrice delle feature
    const X = [];
    for (var i = 0; i < x.length; i++) {
        const row = [];
        for (var j = 0; j <= degree; j++) {
            row.push(Math.pow(x[i], j));
        }
        X.push(row);
    }

    // Calcola i coefficienti del modello di regressione polinomiale
    const coefficients = math.lusolve(X, y);

    return coefficients;
}
// Exponential regression
function exponentialRegression(x, y) {
    const logY = y.map(val => Math.log(val));
    return GiveMe('linearRegression')(x, logY);
}
// Logarithmic regression
function logarithmicRegression(x, y) {
    const logX = x.map(val => Math.log(val));
    return GiveMe('linearRegression')(logX, y);
}
// Power regression
function powerRegression(x, y) {
    const logX = x.map(val => Math.log(val));
    const logY = y.map(val => Math.log(val));
    return GiveMe('linearRegression')(logX, logY);
}
// Matrix manipulation functions
function matrix(rows, cols, initialValue = 0) {
    return Array.from({ length: rows }, () => Array(cols).fill(initialValue));
}

function matrixMultiply(mat1, mat2) {
    const rows1 = mat1.length;
    const cols1 = mat1[0].length;
    const rows2 = mat2.length;
    const cols2 = mat2[0].length;

    if (cols1 !== rows2) {
        throw new Error('Invalid matrix multiplication: incompatible dimensions');
    }

    const result = GiveMe('matrix')(rows1, cols2);

    for (var i = 0; i < rows1; i++) {
        for (var j = 0; j < cols2; j++) {
            for (var k = 0; k < cols1; k++) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return result;
}

function matrixTranspose(mat) {
    const rows = mat.length;
    const cols = mat[0].length;

    const result = GiveMe('matrix')(cols, rows);

    for (var i = 0; i < rows; i++) {
        for (var j = 0; j < cols; j++) {
            result[j][i] = mat[i][j];
        }
    }

    return result;
}

function matrixAdd(mat1, mat2) {
    const rows = mat1.length;
    const cols = mat1[0].length;

    const result = GiveMe('matrix')(rows, cols);

    for (var i = 0; i < rows; i++) {
        for (var j = 0; j < cols; j++) {
            result[i][j] = mat1[i][j] + mat2[i][j];
        }
    }

    return result;
}

function matrixIdentity(size) {
    const result = GiveMe('matrix')(size, size);
    for (var i = 0; i < size; i++) {
        result[i][i] = 1;
    }
    return result;
}

function matrixInverse(matrix) {
    return math.inv(matrix);
}

// Sigmoid function
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// Probabilistic hypothesis testing functions
function t_studentTestProbabilistic(sampleData, populationMean, index = 0, sum = 0) {
    if (index === sampleData.length) {
        const tStat = sum / (sampleData.length - 1) / Math.sqrt(sampleData.length);
        return sigmoid(Math.abs(tStat));
    }

    const currentDiff = sampleData[index] - populationMean;
    return t_studentTestProbabilistic(sampleData, populationMean, index + 1, sum + Math.pow(currentDiff, 2));
}

function gaussianTestProbabilistic(sampleData, index = 0, sum = 0) {
    if (index === sampleData.length) {
        const zScore = sum / Math.sqrt(sampleData.length - 1);
        return sigmoid(Math.abs(zScore));
    }

    const currentVal = sampleData[index];
    const currentDiff = Math.pow(currentVal - mean(sampleData), 2);
    return gaussianTestProbabilistic(sampleData, index + 1, sum + currentDiff);
}

function chiSquareTestProbabilistic(observed, expected, index = 0, sum = 0) {
    if (index === observed.length) {
        const chiSquareStat = sum;
        return sigmoid(chiSquareStat);
    }

    const observedVal = observed[index];
    const expectedVal = expected[index];
    const currentDiff = Math.pow(observedVal - expectedVal, 2) / expectedVal;
    return chiSquareTestProbabilistic(observed, expected, index + 1, sum + currentDiff);
}
// Fourier Transform
function discreteFourierTransform(data) {
    const N = data.length;
    const result = [];
    for (let k = 0; k < N; k++) {
        let real = 0;
        let imag = 0;
        for (let n = 0; n < N; n++) {
            const angle = (2 * Math.PI * k * n) / N;
            real += data[n] * Math.cos(angle);
            imag -= data[n] * Math.sin(angle);
        }
        real /= N;
        imag /= N;
        result.push({ real, imag });
    }
    return result;
}
// Inverse Fourier
function inverseDiscreteFourierTransform(frequencies) {
    const N = frequencies.length;
    const result = [];
    for (let n = 0; n < N; n++) {
        let value = 0;
        for (let k = 0; k < N; k++) {
            const angle = (2 * Math.PI * k * n) / N;
            const real = frequencies[k].real;
            const imag = frequencies[k].imag;
            value += real * Math.cos(angle) - imag * Math.sin(angle);
        }
        result.push(value);
    }
    return result;
}
// Class for Extended Kalman Filter
class ExtendedKalmanFilter {
    constructor(initialState, initialCovariance, processNoise, measurementNoise) {
        this.state = initialState;
        this.covariance = initialCovariance;
        this.processNoise = processNoise;
        this.measurementNoise = measurementNoise;
    }

    // Prediction step
    predict() {
        const F = GiveMe('matrix')([[1, 1], [0, 1]]); // State transition matrix
        const Q = this.processNoise; // Process noise covariance matrix

        // Predict state and covariance
        this.state = GiveMe('matrixMultiply')(F, this.state);
        this.covariance = GiveMe('matrixAdd')(
            GiveMe('matrixMultiply')(GiveMe('matrixMultiply')(F, this.covariance), GiveMe('matrixTranspose')(F)),
            Q
        );
    }

    // Update step
    update(measurement) {
        const H = GiveMe('matrix')([[1, 0]]); // Measurement matrix
        const R = this.measurementNoise; // Measurement noise covariance matrix

        // Calculate innovation
        const y = GiveMe('matrixSubtract')(measurement, GiveMe('matrixMultiply')(H, this.state));

        // Calculate innovation covariance
        const S = GiveMe('matrixAdd')(
            GiveMe('matrixMultiply')(GiveMe('matrixMultiply')(H, this.covariance), GiveMe('matrixTranspose')(H)),
            R
        );

        // Calculate Kalman gain
        const K = GiveMe('matrixMultiply')(GiveMe('matrixMultiply')(this.covariance, GiveMe('matrixTranspose')(H)), GiveMe('matrixInverse')(S));

        // Update state and covariance
        this.state = GiveMe('matrixAdd')(this.state, GiveMe('matrixMultiply')(K, y));
        this.covariance = GiveMe('matrixMultiply')(GiveMe('matrixSubtract')(GiveMe('matrixIdentity')(this.state.length), GiveMe('matrixMultiply')(K, H)), this.covariance);
    }
}
// Gradient Descent algo
function gradientDescent(objectiveFunction, initialGuess, learningRate, maxIterations, tolerance) {
    let currentGuess = initialGuess;
    let currentIteration = 0;
    while (currentIteration < maxIterations) {
        const gradient = computeGradient(objectiveFunction, currentGuess);
        const nextGuess = updateGuess(currentGuess, gradient, learningRate);
        if (convergenceReached(currentGuess, nextGuess, tolerance)) {
            return nextGuess;
        }
        currentGuess = nextGuess;
        currentIteration++;
    }
    return currentGuess;
}

function computeGradient(objectiveFunction, guess) {
    const epsilon = 1e-6; // Small value for numerical stability
    const gradient = [];
    for (let i = 0; i < guess.length; i++) {
        const perturbation = Array(guess.length).fill(0);
        perturbation[i] = epsilon;
        const partialDerivative = (objectiveFunction(guess.map((val, index) => val + perturbation[index])) - objectiveFunction(guess)) / epsilon;
        gradient.push(partialDerivative);
    }
    return gradient;
}

function updateGuess(guess, gradient, learningRate) {
    return guess.map((val, index) => val - learningRate * gradient[index]);
}

function convergenceReached(currentGuess, nextGuess, tolerance) {
    const squaredError = currentGuess.reduce((acc, val, index) => acc + Math.pow(val - nextGuess[index], 2), 0);
    return Math.sqrt(squaredError) < tolerance;
}
// Convolution function
function convolution(signal1, signal2) {
    const resultLength = signal1.length + signal2.length - 1;
    const result = Array(resultLength).fill(0);
    for (let i = 0; i < signal1.length; i++) {
        for (let j = 0; j < signal2.length; j++) {
            result[i + j] += signal1[i] * signal2[j];
        }
    }
    return result;
}
// Principal component analysis
function principalComponentAnalysis(data) {
    const pca = new PCA(data);
    return pca.predict(data);
}
// Cluster analysis (K-Means)
function clusterAnalysis(data, numClusters) {

    // Random centroids init
    let centroids = [];
    for (let i = 0; i < numClusters; i++) {
        centroids.push(data[Math.floor(Math.random() * data.length)]);
    }

    let clusters = [];
    let iterations = 0;
    let maxIterations = 1000; // Max interations

    while (true) {
        // Every point to the next centroid
        clusters = Array.from({ length: numClusters }, () => []);
        for (const point of data) {
            let minDistance = Infinity;
            let closestCluster = -1;
            for (let i = 0; i < numClusters; i++) {
                const distance = Math.abs(point - centroids[i]);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCluster = i;
                }
            }
            clusters[closestCluster].push(point);
        }

        // New centroids calculation
        const newCentroids = [];
        for (const cluster of clusters) {
            const clusterMean = cluster.length > 0 ? cluster.reduce((sum, val) => sum + val, 0) / cluster.length : centroids[i];
            newCentroids.push(clusterMean);
        }

        // Check centroid convegence or max iterations
        if (JSON.stringify(centroids) === JSON.stringify(newCentroids) || iterations >= maxIterations) {
            break;
        }

        centroids = newCentroids;
        iterations++;
    }

    return clusters;
}
// Montecarlo Simulation
function monteCarloSimulation(numThrows) {
    const results = [];
    for (let i = 0; i < numThrows; i++) {
        const result = model(); // insert you're model
        results.push(result);
    }
    return results;
}
// Time series analysis (Moving Average)
function timeSeriesAnalysis(data, windowSize) {
    // Check enough data
    if (data.length === 0 || windowSize <= 0 || windowSize > data.length) {
        throw new Error('Invalid input for time series analysis.');
    }

    const movingAverages = [];
    for (let i = 0; i <= data.length - windowSize; i++) {
        const window = data.slice(i, i + windowSize);
        const average = window.reduce((sum, val) => sum + val, 0) / windowSize;
        movingAverages.push(average);
    }

    return movingAverages;
}

// Frequency analysis
function frequencyAnalysis(data) {

    // Uniques frequency calculation 
    const frequencies = data.reduce((freq, value) => {
        freq[value] = (freq[value] || 0) + 1;
        return freq;
    }, {});

    return frequencies;
}

// Data preprocessing 


// Min-Max Scaling
function minMaxScaling(data, minVal, maxVal) {
    const scaledData = data.map((val) => (val - minVal) / (maxVal - minVal));
    return scaledData;
}

// Standardization (Z-score Scaling)
function standardization(data, mean, stdDev) {
    const standardizedData = data.map((val) => (val - mean) / stdDev);
    return standardizedData;
}

// Normalization
function normalization(data) {
    const magnitude = Math.sqrt(data.reduce((acc, val) => acc + val ** 2, 0));
    const normalizedData = data.map((val) => val / magnitude);
    return normalizedData;
}

// Log Transformation
function logTransformation(data) {
    const transformedData = data.map((val) => Math.log(val));
    return transformedData;
}

// Box-Cox Transformation // lambda == 1 == logarithmic?
function boxCoxTransformation(data, lambda) {
    const transformedData = data.map((val) => (Math.pow(val, lambda) - 1) / lambda);
    return transformedData;
}

// Feature Scaling
function featureScaling(data, targetMean, targetStdDev) {
    const meanData = GiveMe('mean')(data);
    const stdDevData = GiveMe('standardDeviation')(data);
    const scaledData = data.map((val) => targetMean + (val - meanData) * (targetStdDev / stdDevData));
    return scaledData;
}
// Distances definition // Clarification : n-dimensionality points

function euclideanDistance(point1, point2) {
    let sumOfSquares = 0;
    for (let i = 0; i < point1.length; i++) {
        sumOfSquares += Math.pow(point1[i] - point2[i], 2);
    }
    return Math.sqrt(sumOfSquares);
}
function manhattanDistance(point1, point2) {
    let sumOfDiffs = 0;
    for (let i = 0; i < point1.length; i++) {
        sumOfDiffs += Math.abs(point1[i] - point2[i]);
    }
    return sumOfDiffs;
}
function chebyshevDistance(point1, point2) {
    let maxDiff = 0;
    for (let i = 0; i < point1.length; i++) {
        const diff = Math.abs(point1[i] - point2[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
    }
    return maxDiff;
}

// K-Nearest Neighbors (KNN)
function kNearestNeighbors(trainingData, queryPoint, k, distanceMeasure) {   //Euclidean distance? Manhattan distance?
    // Calculate distances between query point and all training data points
    const distances = [];
    for (const dataPoint of trainingData) {
        const distance = distanceMeasure(queryPoint, dataPoint);
        distances.push({ point: dataPoint, distance });
    }

    // Sort distances in ascending order
    distances.sort((a, b) => a.distance - b.distance);

    // Select the top k nearest neighbors
    const nearestNeighbors = distances.slice(0, k);

    // Return the nearest neighbors
    return nearestNeighbors;
}

// Exporting functions
module.exports = {
    GiveMe
};
