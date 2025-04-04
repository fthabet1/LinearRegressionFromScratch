// Function to load CSV datasets
async function loadDatasetFromCSV(filename) {
    try {
        const response = await fetch(`/datasets/${filename}`);
        if (!response.ok) {
            throw new Error(`Failed to load dataset: ${response.statusText}`);
        }
        
        const csvText = await response.text();
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        
        // Determine feature and target columns
        const featureColumns = headers.slice(0, -1);
        const targetColumn = headers[headers.length - 1];
        
        const data = {
            X: [],
            y: [],
            xLabel: featureColumns.join(', '),
            yLabel: targetColumn
        };
        
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => parseFloat(v.trim()));
            
            // Skip lines with NaN values
            if (!values.some(isNaN)) {
                if (featureColumns.length === 1) {
                    // Univariate case
                    data.X.push([values[0]]);
                } else {
                    // Multivariate case
                    data.X.push(values.slice(0, -1));
                }
                data.y.push(values[values.length - 1]);
            }
        }
        
        return data;
    } catch (error) {
        console.error('Error loading dataset:', error);
        throw error;
    }
}

// Function to split data into train and test sets
function trainTestSplit(data, testSize = 0.2) {
    // Calculate the number of samples for test set
    const numSamples = data.X.length;
    const testCount = Math.floor(numSamples * testSize);
    
    // Create random indices for shuffling
    const indices = [...Array(numSamples).keys()];
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]]; // Swap elements
    }
    
    // Split indices into train and test
    const testIndices = indices.slice(0, testCount);
    const trainIndices = indices.slice(testCount);
    
    // Create train and test datasets
    const trainData = {
        X: trainIndices.map(i => data.X[i]),
        y: trainIndices.map(i => data.y[i]),
        xLabel: data.xLabel,
        yLabel: data.yLabel
    };
    
    const testData = {
        X: testIndices.map(i => data.X[i]),
        y: testIndices.map(i => data.y[i]),
        xLabel: data.xLabel,
        yLabel: data.yLabel
    };
    
    return { trainData, testData };
}

// Function for k-fold cross-validation using the backend DataSplitter
async function crossValidate(model, data, k = 5) {
    if (!data || !data.X || data.X.length === 0) {
        throw new Error('No data provided for cross-validation');
    }
    
    try {
        // Convert data to the format expected by the backend
        const trainData = {
            X: data.X,
            y: data.y
        };
        
        // Call the backend API to perform cross-validation
        const response = await fetch('/api/cross_validate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: trainData,
                model_type: model.modelType,
                learning_rate: parseFloat(learningRate.value),
                max_iterations: parseInt(maxIterations.value),
                lambda: parseFloat(lambdaInput.value),
                n_folds: k
            })
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${errorText}`);
        }
        
        const cvResults = await response.json();
        
        if (!cvResults.success) {
            throw new Error(cvResults.error || 'Cross-validation failed');
        }
        
        console.log("Cross-validation results from backend:", cvResults);
        
        // After cross-validation, the backend has also trained the final model
        // on the entire training dataset, so we don't need to do it again
        
        return {
            avgR2: cvResults.avg_r2,
            avgMSE: cvResults.avg_mse,
            foldMetrics: cvResults.fold_metrics,
            finalModel: cvResults.final_model
        };
    } catch (error) {
        console.error("Error in cross-validation:", error);
        throw error;
    }
}

// Dataset mapping
const datasetMapping = {
    'univariateStudentData': {
        file: 'univariateStudentData.csv',
        type: 'univariate',
        name: 'Student Performance',
        sampleCount: 397
    },
    'univariateIceCreamData': {
        file: 'univariateIceCreamData.csv',
        type: 'univariate',
        name: 'Ice Cream Sales',
        sampleCount: 367
    },
    'univariateSalaryData': {
        file: 'univariateSalaryData.csv',
        type: 'univariate',
        name: 'Salary vs Experience',
        sampleCount: 32
    },
    'multivariateStudentData': {
        file: 'multivariateStudentData.csv',
        type: 'multivariate',
        name: 'Student Performance (Multi)',
        sampleCount: 10000
    },
    'multivariateCarPricesData': {
        file: 'multivariateCarPricesData.csv',
        type: 'multivariate',
        name: 'Car Prices',
        sampleCount: 2097
    },
    'multivariateHousingData': {
        file: 'multivariateHousingData.csv',
        type: 'multivariate',
        name: 'Housing Prices',
        sampleCount: 10000
    }
};

let currentModel = null;
let chart = null;
let currentData = null;
let trainData = null;
let testData = null;
let trainPredictions = null;
let testPredictions = null;
let viewMode = 'train'; // 'train' or 'test' or 'both'

// DOM Elements
const modelType = document.getElementById('modelType');
const learningRate = document.getElementById('learningRate');
const maxIterations = document.getElementById('maxIterations');
const lambdaInput = document.getElementById('lambda');
const datasetSelect = document.getElementById('datasetSelect');
const trainButton = document.getElementById('trainButton');
const predictButton = document.getElementById('predictButton');
const testModeToggle = document.getElementById('testModeToggle');
const r2ScoreElement = document.getElementById('r2Score');
const mseElement = document.getElementById('mse');
const r2ScoreTestElement = document.getElementById('r2ScoreTest');
const mseTestElement = document.getElementById('mseTest');
const r2ScorePredTrainElement = document.getElementById('r2ScorePredTrain');
const msePredTrainElement = document.getElementById('msePredTrain');
const r2ScorePredTestElement = document.getElementById('r2ScorePredTest');
const msePredTestElement = document.getElementById('msePredTest');
const samplingInfoElement = document.getElementById('samplingInfo');

// Initialize Chart.js
function initChart() {
    const ctx = document.getElementById('plotCanvas').getContext('2d');
    chart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Training Data',
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    data: []
                },
                {
                    label: 'Testing Data',
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    data: [],
                    hidden: true
                },
                {
                    label: 'Training Predictions',
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    data: [],
                    type: 'line',
                    tension: 0.1
                }
                // A fourth dataset for test predictions will be added dynamically when needed
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Feature'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Target'
                    }
                }
            },
            animation: {
                duration: 1000
            }
        }
    });
}

// Create model based on selected type
function createModel() {
    const type = modelType.value;
    const lr = parseFloat(learningRate.value);
    const iterations = parseInt(maxIterations.value);
    const lambda = parseFloat(lambdaInput.value);

    switch (type) {
        case 'univariate':
            return new UnivariateLinearModel(lr, iterations);
        case 'multivariate':
            return new MultivariateLinearModel(lr, iterations);
        case 'ridge':
            return new RidgeRegression(lr, iterations, lambda);
        case 'lasso':
            return new LassoRegression(lr, iterations, lambda);
        default:
            throw new Error('Invalid model type');
    }
}

// Toggle between train and test view
function toggleViewMode() {
    if (viewMode === 'train') {
        viewMode = 'test';
    } else if (viewMode === 'test') {
        viewMode = 'both';
    } else {
        viewMode = 'train';
    }
    updateVisualization();
    
    if (viewMode === 'train') {
        testModeToggle.textContent = 'View Test Data';
    } else if (viewMode === 'test') {
        testModeToggle.textContent = 'View Both';
    } else {
        testModeToggle.textContent = 'View Training Data';
    }
}

// Update visualization based on current view mode
function updateVisualization() {
    if (!trainData || !testData) return;
    
    // Show/hide datasets based on view mode
    chart.data.datasets[0].hidden = viewMode !== 'train' && viewMode !== 'both'; // Training data
    chart.data.datasets[1].hidden = viewMode !== 'test' && viewMode !== 'both';  // Test data
    
    // Update predictions lines
    if (trainPredictions && (viewMode === 'train' || viewMode === 'both')) {
        updatePredictionLine(trainData, trainPredictions, 2, 'rgba(255, 99, 132, 1)');
    }
    
    if (testPredictions && (viewMode === 'test' || viewMode === 'both')) {
        // If showing both, we need to ensure we have separate prediction lines
        if (viewMode === 'both') {
            // Make sure we have a test prediction line (dataset index 3)
            if (chart.data.datasets.length < 4) {
                chart.data.datasets.push({
                    label: 'Test Predictions',
                    backgroundColor: 'rgba(153, 102, 255, 0.5)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    data: [],
                    type: 'line',
                    tension: 0.1,
                    borderDash: [5, 5]  // Dashed line for test predictions
                });
            }
            updatePredictionLine(testData, testPredictions, 3, 'rgba(153, 102, 255, 1)');
        } else {
            updatePredictionLine(testData, testPredictions, 2, 'rgba(255, 99, 132, 1)');
        }
    }
    
    chart.update();
}

// Update chart with new data
function updateChart(fullData) {
    if (!chart || !fullData || !fullData.X || fullData.X.length === 0) return;
    
    // Split data into train and test
    const { trainData: newTrainData, testData: newTestData } = trainTestSplit(fullData);
    trainData = newTrainData;
    testData = newTestData;
    
    // For multivariate data, we'll plot against the first feature
    const featureIdx = 0;
    
    // Function to sample data points for display
    function sampleDataPoints(data, maxPoints) {
        if (data.X.length <= maxPoints) {
            return Array.from({ length: data.X.length }, (_, i) => i);
        }
        
        const step = Math.floor(data.X.length / maxPoints);
        return Array.from({ length: maxPoints }, (_, i) => i * step)
            .filter(idx => idx < data.X.length);
    }
    
    // Limit points for better performance
    const maxPointsToDisplay = 500;
    const trainDisplayIndices = sampleDataPoints(trainData, maxPointsToDisplay);
    const testDisplayIndices = sampleDataPoints(testData, maxPointsToDisplay);
    
    // Show sampling info if we're sampling
    samplingInfoElement.style.display = 
        (trainData.X.length > maxPointsToDisplay || testData.X.length > maxPointsToDisplay) ? 
        'block' : 'none';
    
    // Update training data points
    chart.data.datasets[0].data = trainDisplayIndices.map(i => ({
        x: trainData.X[i][featureIdx],
        y: trainData.y[i]
    }));
    
    // Update test data points
    chart.data.datasets[1].data = testDisplayIndices.map(i => ({
        x: testData.X[i][featureIdx],
        y: testData.y[i]
    }));
    
    // Clear prediction lines
    chart.data.datasets[2].data = [];
    if (chart.data.datasets.length > 3) {
        chart.data.datasets[3].data = [];
    }
    
    // Update axis labels
    chart.options.scales.x.title.text = fullData.xLabel.split(',')[0] || 'Feature';
    chart.options.scales.y.title.text = fullData.yLabel || 'Target';
    
    // Update visibility based on view mode
    updateVisualization();
    
    chart.update();
}

// Update prediction line
function updatePredictionLine(data, predictions, datasetIndex, color) {
    if (!chart || !data || !predictions) return;
    
    const featureIdx = 0;
    
    // For the line, use more points to ensure a smooth curve
    const maxLinePoints = 200;
    
    // Create indices of all points
    const indices = Array.from({ length: data.X.length }, (_, i) => i);
    
    // Sort by x value for a smooth line
    const sortedIndices = indices.sort((a, b) => data.X[a][featureIdx] - data.X[b][featureIdx]);
    
    // Sample points if needed
    let lineIndices;
    if (sortedIndices.length > maxLinePoints) {
        const lineStep = Math.floor(sortedIndices.length / maxLinePoints);
        lineIndices = Array.from({ length: maxLinePoints }, (_, i) => sortedIndices[i * lineStep])
            .filter(idx => idx !== undefined);
    } else {
        lineIndices = sortedIndices;
    }
    
    // Update prediction line
    chart.data.datasets[datasetIndex].data = lineIndices.map(i => ({
        x: data.X[i][featureIdx],
        y: predictions[i]
    }));
    
    if (color) {
        chart.data.datasets[datasetIndex].borderColor = color;
    }
}

// Calculate metrics (R² score and MSE)
async function calculateMetrics(X, y_true, y_pred) {
    try {
        // If y_pred is the same as y_true, this is a baseline comparison
        const isPerfectPrediction = y_true === y_pred;
        
        // For baseline metrics (when comparing original data to itself)
        if (isPerfectPrediction) {
            return {
                r2: 1.0,  // Perfect R² score is 1.0
                mse: 0.0  // Perfect MSE is 0.0
            };
        }
        
        // For model predictions
        // If we have a model trained, use the model's score method for R²
        let r2 = 0;
        if (currentModel && currentModel.score) {
            r2 = await currentModel.score(X, y_true);
        } else {
            // Fallback to a simplified R² calculation
            const mean = y_true.reduce((sum, val) => sum + val, 0) / y_true.length;
            const totalSS = y_true.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);
            const residualSS = y_true.reduce((sum, val, i) => sum + Math.pow(val - y_pred[i], 2), 0);
            r2 = 1 - (residualSS / totalSS);
            // Cap R² at 0 (can be negative if model performs worse than mean baseline)
            r2 = Math.max(0, r2);
        }
        
        // Calculate Mean Squared Error
        const mse = y_true.reduce((sum, val, i) => sum + Math.pow(val - y_pred[i], 2), 0) / y_true.length;
        
        return { r2, mse };
    } catch (error) {
        console.error("Error calculating metrics:", error);
        return { r2: 0, mse: 0 };
    }
}

// Update dataset options based on model type
function updateDatasetOptions(selectedModelType) {
    // Clear existing options first
    while (datasetSelect.options.length > 1) {
        datasetSelect.remove(1);
    }
    
    // Create option groups
    let univariateGroup = datasetSelect.querySelector('optgroup[label="Univariate Datasets"]');
    let multivariateGroup = datasetSelect.querySelector('optgroup[label="Multivariate Datasets"]');
    
    // If they don't exist, create them
    if (!univariateGroup) {
        univariateGroup = document.createElement('optgroup');
        univariateGroup.label = "Univariate Datasets";
        datasetSelect.appendChild(univariateGroup);
    } else {
        // Clear existing options
        while (univariateGroup.firstChild) {
            univariateGroup.removeChild(univariateGroup.firstChild);
        }
    }
    
    if (!multivariateGroup) {
        multivariateGroup = document.createElement('optgroup');
        multivariateGroup.label = "Multivariate Datasets";
        datasetSelect.appendChild(multivariateGroup);
    } else {
        // Clear existing options
        while (multivariateGroup.firstChild) {
            multivariateGroup.removeChild(multivariateGroup.firstChild);
        }
    }
    
    // Add appropriate dataset options based on selected model type
    Object.entries(datasetMapping).forEach(([key, dataset]) => {
        // Determine if the dataset should be shown based on model type
        let showDataset = false;
        
        if (selectedModelType === 'univariate') {
            // Univariate model can only use univariate datasets
            showDataset = dataset.type === 'univariate';
        } else {
            // Multivariate, Ridge, and Lasso models should only use multivariate datasets
            showDataset = dataset.type === 'multivariate';
        }
        
        if (showDataset) {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = `${dataset.name} (${dataset.sampleCount} samples)`;
            
            if (dataset.type === 'univariate') {
                univariateGroup.appendChild(option);
            } else {
                multivariateGroup.appendChild(option);
            }
        }
    });
    
    // Hide groups if they have no children
    univariateGroup.style.display = univariateGroup.children.length > 0 ? '' : 'none';
    multivariateGroup.style.display = multivariateGroup.children.length > 0 ? '' : 'none';
    
    // Reset dataset selection
    datasetSelect.value = '';
    currentData = null;
    trainData = null;
    testData = null;
    trainPredictions = null;
    testPredictions = null;
    viewMode = 'train';  // Reset view mode
    
    // Reset chart and metrics
    if (chart) {
        chart.data.datasets[0].data = [];
        chart.data.datasets[1].data = [];
        chart.data.datasets[2].data = [];
        if (chart.data.datasets.length > 3) {
            chart.data.datasets[3].data = [];
        }
        chart.update();
    }
    
    // Reset all metrics values
    resetAllMetrics();
    
    // Clear any highlighting on prediction metrics cards
    const trainMetricsCard = r2ScorePredTrainElement.closest('.card');
    const testMetricsCard = r2ScorePredTestElement.closest('.card');
    if (trainMetricsCard && testMetricsCard) {
        trainMetricsCard.style.border = '';
        testMetricsCard.style.border = '';
        trainMetricsCard.title = '';
        testMetricsCard.title = '';
    }
    
    // Disable buttons
    predictButton.disabled = true;
    if (testModeToggle) testModeToggle.disabled = true;
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    updateDatasetOptions(modelType.value);

    // Initialize lambda visibility
    const isRegularized = ['ridge', 'lasso'].includes(modelType.value);
    lambdaInput.parentElement.style.display = isRegularized ? 'block' : 'none';
    
    // Initialize test mode toggle button if it exists
    if (testModeToggle) {
        testModeToggle.addEventListener('click', toggleViewMode);
        testModeToggle.disabled = true;
    }
});

modelType.addEventListener('change', () => {
    const isRegularized = ['ridge', 'lasso'].includes(modelType.value);
    lambdaInput.parentElement.style.display = isRegularized ? 'block' : 'none';
    
    // Update dataset options based on model type
    updateDatasetOptions(modelType.value);
});

datasetSelect.addEventListener('change', async () => {
    if (!datasetSelect.value) return;
    
    try {
        const datasetInfo = datasetMapping[datasetSelect.value];
        if (!datasetInfo) {
            throw new Error('Dataset not found');
        }
        
        trainButton.disabled = true;
        trainButton.textContent = 'Loading dataset...';
        
        currentData = await loadDatasetFromCSV(datasetInfo.file);
        console.log("Loaded dataset:", currentData);
        
        // Update chart with split data
        updateChart(currentData);
        
        // Check if current model type is compatible with the dataset
        const isUnivariate = modelType.value === 'univariate';
        const dataIsMultivariate = currentData.X[0].length > 1;
        
        if (isUnivariate && dataIsMultivariate) {
            alert('Switching to multivariate model for multi-feature dataset.');
            modelType.value = 'multivariate';
            const isRegularized = ['ridge', 'lasso'].includes(modelType.value);
            lambdaInput.parentElement.style.display = isRegularized ? 'block' : 'none';
        }
        
        // Reset all metrics
        resetAllMetrics();
        
        // Clear any highlighting on prediction metrics cards
        const trainMetricsCard = r2ScorePredTrainElement.closest('.card');
        const testMetricsCard = r2ScorePredTestElement.closest('.card');
        trainMetricsCard.style.border = '';
        testMetricsCard.style.border = '';
        trainMetricsCard.title = '';
        testMetricsCard.title = '';
        
        predictButton.disabled = true;
        if (testModeToggle) testModeToggle.disabled = true;
        
        trainButton.disabled = false;
        trainButton.textContent = 'Train Model';
        
    } catch (error) {
        console.error("Error loading dataset:", error);
        alert('Error loading dataset: ' + error.message);
        datasetSelect.value = '';
        
        trainButton.disabled = false;
        trainButton.textContent = 'Train Model';
    }
});

// Function to reset all metrics to '-'
function resetAllMetrics() {
    // Reset train/test split metrics
    r2ScoreElement.textContent = '-';
    mseElement.textContent = '-';
    r2ScoreTestElement.textContent = '-';
    mseTestElement.textContent = '-';
    
    // Reset prediction metrics
    r2ScorePredTrainElement.textContent = '-';
    msePredTrainElement.textContent = '-';
    r2ScorePredTestElement.textContent = '-';
    msePredTestElement.textContent = '-';
}

trainButton.addEventListener('click', async () => {
    try {
        if (!currentData || !trainData || !testData) {
            throw new Error('Please select a dataset');
        }

        // Create model
        currentModel = createModel();
        console.log("Starting cross-validation with training data:", trainData);
        
        // Add loading state
        trainButton.disabled = true;
        trainButton.textContent = 'Cross-validating...';
        
        // Perform cross-validation on the training data only using the backend
        const cvResults = await crossValidate(currentModel, trainData, 5);
        
        console.log("Cross-validation complete:", cvResults);
        
        // Update the UI with cross-validation results
        r2ScoreElement.textContent = cvResults.avgR2.toFixed(4);
        mseElement.textContent = cvResults.avgMSE.toFixed(2);
        
        // For the test metrics, use placeholder values until the user clicks "Make Predictions"
        r2ScoreTestElement.textContent = '-';
        mseTestElement.textContent = '-';
        
        // Reset prediction metrics
        r2ScorePredTrainElement.textContent = '-';
        msePredTrainElement.textContent = '-';
        r2ScorePredTestElement.textContent = '-';
        msePredTestElement.textContent = '-';
        
        // Remove any highlighting
        const trainMetricsCard = r2ScoreElement.closest('.card');
        const testMetricsCard = r2ScoreTestElement.closest('.card');
        trainMetricsCard.style.border = '';
        testMetricsCard.style.border = '';
        trainMetricsCard.title = '';
        testMetricsCard.title = '';
        
        const trainPredMetricsCard = r2ScorePredTrainElement.closest('.card');
        const testPredMetricsCard = r2ScorePredTestElement.closest('.card');
        trainPredMetricsCard.style.border = '';
        testPredMetricsCard.style.border = '';
        trainPredMetricsCard.title = '';
        testPredMetricsCard.title = '';

        // Set view mode to training data
        viewMode = 'train';
        updateVisualization();
        if (testModeToggle) testModeToggle.textContent = 'View Test Data';

        // Enable buttons
        predictButton.disabled = false;
        if (testModeToggle) testModeToggle.disabled = false;
        
        // Reset train button
        trainButton.disabled = false;
        trainButton.textContent = 'Train Model';

    } catch (error) {
        console.error("Training error:", error);
        alert('Error: ' + error.message);
        
        // Reset train button
        trainButton.disabled = false;
        trainButton.textContent = 'Train Model';
    }
});

// Function to highlight performance difference between training and test metrics for the split data
function highlightSplitPerformanceDifference(trainR2, testR2) {
    const trainMetricsCard = r2ScoreElement.closest('.card');
    const testMetricsCard = r2ScoreTestElement.closest('.card');
    
    // Reset any previous highlighting
    trainMetricsCard.style.border = '';
    testMetricsCard.style.border = '';
    
    // Add metrics-card class for hover effect
    trainMetricsCard.classList.add('metrics-card');
    testMetricsCard.classList.add('metrics-card');
    
    // Calculate difference and apply visual feedback
    const difference = trainR2 - testR2;
    
    if (Math.abs(difference) < 0.05) {
        // Good - model generalizes well
        trainMetricsCard.style.border = '1px solid rgba(40, 167, 69, 0.5)';
        testMetricsCard.style.border = '1px solid rgba(40, 167, 69, 0.5)';
        
        // Add tooltip information
        trainMetricsCard.title = 'Good model fit on training data';
        testMetricsCard.title = 'Good generalization on test data';
    } else if (difference > 0.2) {
        // Significant overfitting
        trainMetricsCard.style.border = '1px solid rgba(220, 53, 69, 0.5)';
        testMetricsCard.style.border = '1px solid rgba(220, 53, 69, 0.5)';
        
        // Add tooltip information
        trainMetricsCard.title = 'Potential overfitting: Model performs much better on training data';
        testMetricsCard.title = 'Poor generalization: Consider regularization to improve test performance';
    } else {
        // Some overfitting, but not extreme
        trainMetricsCard.style.border = '1px solid rgba(255, 193, 7, 0.5)';
        testMetricsCard.style.border = '1px solid rgba(255, 193, 7, 0.5)';
        
        // Add tooltip information
        trainMetricsCard.title = 'Some overfitting: Model performs better on training data';
        testMetricsCard.title = 'Moderate generalization: Consider tuning model parameters';
    }
}

// Function to highlight performance difference between prediction metrics
function highlightPerformanceDifference(trainR2, testR2) {
    const trainMetricsCard = r2ScorePredTrainElement.closest('.card');
    const testMetricsCard = r2ScorePredTestElement.closest('.card');
    
    // Reset any previous highlighting
    trainMetricsCard.style.border = '';
    testMetricsCard.style.border = '';
    
    // Add metrics-card class for hover effect
    trainMetricsCard.classList.add('metrics-card');
    testMetricsCard.classList.add('metrics-card');
    
    // Calculate difference and apply visual feedback
    const difference = trainR2 - testR2;
    
    if (Math.abs(difference) < 0.05) {
        // Good - model generalizes well
        trainMetricsCard.style.border = '1px solid rgba(40, 167, 69, 0.5)';
        testMetricsCard.style.border = '1px solid rgba(40, 167, 69, 0.5)';
        
        // Add tooltip information
        trainMetricsCard.title = 'Good model fit: Training and testing performance are similar';
        testMetricsCard.title = 'Good generalization: Model performs well on unseen data';
    } else if (difference > 0.2) {
        // Significant overfitting
        trainMetricsCard.style.border = '1px solid rgba(220, 53, 69, 0.5)';
        testMetricsCard.style.border = '1px solid rgba(220, 53, 69, 0.5)';
        
        // Add tooltip information
        trainMetricsCard.title = 'Potential overfitting: Model performs much better on training data';
        testMetricsCard.title = 'Poor generalization: Model performs worse on unseen data';
    } else {
        // Some overfitting, but not extreme
        trainMetricsCard.style.border = '1px solid rgba(255, 193, 7, 0.5)';
        testMetricsCard.style.border = '1px solid rgba(255, 193, 7, 0.5)';
        
        // Add tooltip information
        trainMetricsCard.title = 'Some overfitting: Model performs better on training data';
        testMetricsCard.title = 'Moderate generalization: Consider adjusting regularization';
    }
}

predictButton.addEventListener('click', async () => {
    try {
        if (!currentModel || !trainData || !testData) {
            throw new Error('Please train the model first');
        }

        // Make predictions on both train and test datasets
        trainPredictions = await currentModel.predict(trainData.X);
        testPredictions = await currentModel.predict(testData.X);
        
        // Update visualization
        updateVisualization();

        // Calculate and display metrics for training predictions
        const trainPredMetrics = await calculateMetrics(trainData.X, trainData.y, trainPredictions);
        r2ScorePredTrainElement.textContent = trainPredMetrics.r2.toFixed(4);
        msePredTrainElement.textContent = trainPredMetrics.mse.toFixed(2);
        
        // Calculate and display metrics for test predictions (this is the first time we use the test set)
        const testPredMetrics = await calculateMetrics(testData.X, testData.y, testPredictions);
        r2ScorePredTestElement.textContent = testPredMetrics.r2.toFixed(4);
        msePredTestElement.textContent = testPredMetrics.mse.toFixed(2);
        
        // Update the test metrics in the train/test split section as well
        r2ScoreTestElement.textContent = testPredMetrics.r2.toFixed(4);
        mseTestElement.textContent = testPredMetrics.mse.toFixed(2);
        
        // Highlight performance difference
        highlightPerformanceDifference(trainPredMetrics.r2, testPredMetrics.r2);
        
        // Also highlight the train/test split cards
        highlightSplitPerformanceDifference(parseFloat(r2ScoreElement.textContent), testPredMetrics.r2);
        
        // Show both datasets
        viewMode = 'both';
        updateVisualization();
        if (testModeToggle) testModeToggle.textContent = 'View Training Data';
        
    } catch (error) {
        console.error("Prediction error:", error);
        alert('Error: ' + error.message);
    }
}); 