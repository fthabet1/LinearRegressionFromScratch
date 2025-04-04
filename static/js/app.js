// Function to load CSV datasets using the backend API
async function loadDatasetFromCSV(filename) {
    try {
        console.log(`Loading dataset ${filename} via API`);
        
        const response = await fetch('/api/load_dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: filename
            })
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to load dataset: ${errorText}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Dataset loading failed');
        }
        
        console.log(`Dataset ${filename} loaded successfully`);
        return result.data;
    } catch (error) {
        console.error('Error loading dataset:', error);
        throw error;
    }
}

// Function to split data into train and test sets using the backend API
async function trainTestSplit(data, testSize = 0.2) {
    try {
        console.log(`Splitting data with test_size=${testSize} via API`);
        
        const response = await fetch('/api/train_test_split', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                X: data.X,
                y: data.y,
                xLabel: data.xLabel,
                yLabel: data.yLabel,
                test_size: testSize,
                random_state: 42 // Using a fixed random state for reproducibility
            })
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to split data: ${errorText}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Data splitting failed');
        }
        
        console.log('Data split successfully');
        return {
            trainData: result.trainData,
            testData: result.testData
        };
    } catch (error) {
        console.error('Error splitting data:', error);
        throw error;
    }
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
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Cross-validation failed');
        }
        
        console.log("Raw cross-validation results from backend:", result);
        
        // Transform the response to match our frontend's expected structure
        const cvResults = {
            avgR2: result.avg_r2,
            avgMSE: result.avg_mse,
            avg_training_time: result.avg_training_time,
            avg_iterations: result.avg_iterations,
            foldMetrics: result.fold_metrics,
            finalModel: {
                ...result.final_model,
                // Ensure optimizer_info is properly passed through
                optimizer_info: result.final_model?.optimizer_info || {}
            }
        };
        
        console.log("Transformed cross-validation results:", cvResults);
        
        // After cross-validation, the backend has also trained the final model
        // on the entire training dataset, so we don't need to do it again
        
        return cvResults;
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
const r2ScorePredTestElement = document.getElementById('r2ScorePredTest');
const msePredTestElement = document.getElementById('msePredTest');
const samplingInfoElement = document.getElementById('samplingInfo');
const trainTimeElement = document.getElementById('trainTime');
const iterationsElement = document.getElementById('iterations');
const costImprovementElement = document.getElementById('costImprovement');

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
                    tension: 0.1,
                    hidden: true
                },
                {
                    label: 'Testing Predictions',
                    backgroundColor: 'rgba(153, 102, 255, 0.5)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    data: [],
                    type: 'line',
                    tension: 0.1,
                    borderDash: [5, 5],
                    hidden: true
                }
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
        testModeToggle.textContent = 'View Both';
    } else if (viewMode === 'test') {
        viewMode = 'both';
        testModeToggle.textContent = 'View Training Data';
    } else {
        viewMode = 'train';
        testModeToggle.textContent = 'View Test Data';
    }
    
    console.log("Toggled view mode to:", viewMode);
    updateVisualization();
}

// Update visualization based on current view mode
function updateVisualization() {
    if (!trainData || !testData) return;
    
    console.log("Updating visualization with view mode:", viewMode);
    console.log("Train predictions exists:", !!trainPredictions);
    console.log("Test predictions exists:", !!testPredictions);
    
    // Show/hide datasets based on view mode
    chart.data.datasets[0].hidden = viewMode !== 'train' && viewMode !== 'both'; // Training data
    chart.data.datasets[1].hidden = viewMode !== 'test' && viewMode !== 'both';  // Test data
    
    // Ensure we have a separate dataset for test predictions if it doesn't exist
    if (chart.data.datasets.length < 4) {
        console.log("Adding test prediction dataset to chart");
        chart.data.datasets.push({
            label: 'Test Predictions',
            backgroundColor: 'rgba(153, 102, 255, 0.5)',
            borderColor: 'rgba(153, 102, 255, 1)',
            data: [],
            type: 'line',
            tension: 0.1,
            borderDash: [5, 5],  // Dashed line for test predictions
            hidden: true
        });
    }
    
    // Update training prediction line if it exists (always dataset 2)
    if (trainPredictions) {
        console.log("Updating training prediction line");
        updatePredictionLine(trainData, trainPredictions, 2, 'rgba(255, 99, 132, 1)');
        // Show/hide based on view mode
        chart.data.datasets[2].hidden = viewMode !== 'train' && viewMode !== 'both';
    } else {
        // No training predictions yet
        chart.data.datasets[2].hidden = true;
    }
    
    // Update test prediction line if it exists (always dataset 3)
    if (testPredictions) {
        console.log("Updating test prediction line");
        updatePredictionLine(testData, testPredictions, 3, 'rgba(153, 102, 255, 1)');
        // Show/hide based on view mode
        chart.data.datasets[3].hidden = viewMode !== 'test' && viewMode !== 'both';
    } else {
        // No test predictions yet
        chart.data.datasets[3].hidden = true;
    }
    
    // Log the current state of chart datasets for debugging
    console.log("Chart datasets state:");
    chart.data.datasets.forEach((dataset, i) => {
        console.log(`Dataset ${i}: ${dataset.label}, hidden: ${dataset.hidden}, points: ${dataset.data.length}`);
    });
    
    chart.update();
}

// Update chart with new data
async function updateChart(fullData) {
    if (!chart || !fullData || !fullData.X || fullData.X.length === 0) return;
    
    try {
        // Reset predictions
        trainPredictions = null;
        testPredictions = null;
        
        // Split data into train and test using the API
        const { trainData: newTrainData, testData: newTestData } = await trainTestSplit(fullData);
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
        chart.data.datasets[3].data = [];
        
        // Hide prediction lines until we have predictions
        chart.data.datasets[2].hidden = true;
        chart.data.datasets[3].hidden = true;
        
        // Update axis labels
        chart.options.scales.x.title.text = fullData.xLabel.split(',')[0] || 'Feature';
        chart.options.scales.y.title.text = fullData.yLabel || 'Target';
        
        // Update visibility based on view mode
        updateVisualization();
        
        chart.update();
        
        console.log('Chart updated with train/test data');
    } catch (error) {
        console.error('Error updating chart:', error);
        alert('Error updating chart: ' + error.message);
    }
}

// Update prediction line
function updatePredictionLine(data, predictions, datasetIndex, color) {
    if (!chart || !data || !predictions) {
        console.error("Missing required data for updatePredictionLine:", {
            hasChart: !!chart,
            hasData: !!data,
            hasPredictions: !!predictions
        });
        return;
    }
    
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
    
    console.log(`Updating prediction line for dataset ${datasetIndex}, points: ${lineIndices.length}`);
    
    // Ensure predictions array has values for all indices
    if (Math.max(...lineIndices) >= predictions.length) {
        console.error("Prediction index out of bounds:", {
            maxIndex: Math.max(...lineIndices),
            predictionsLength: predictions.length
        });
        // Limit indices to valid range
        lineIndices = lineIndices.filter(i => i < predictions.length);
    }
    
    // Update prediction line
    const lineData = lineIndices.map(i => ({
        x: data.X[i][featureIdx],
        y: predictions[i]
    }));
    
    // Verify we have valid data points
    const hasInvalidPoints = lineData.some(point => 
        isNaN(point.x) || isNaN(point.y) || 
        point.x === undefined || point.y === undefined
    );
    
    if (hasInvalidPoints) {
        console.error("Invalid points detected in prediction line data");
    } else {
        console.log(`Generated ${lineData.length} valid points for prediction line`);
    }
    
    chart.data.datasets[datasetIndex].data = lineData;
    
    if (color) {
        chart.data.datasets[datasetIndex].borderColor = color;
    }
    
    // Explicitly set visibility
    chart.data.datasets[datasetIndex].hidden = false;
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
        
        // Call backend API for R² calculation
        const r2Response = await fetch('/api/calculate_r2', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                y_true: y_true,
                y_pred: y_pred
            })
        });
        
        const r2Data = await r2Response.json();
        if (!r2Data.success) {
            console.error("Error calculating R²:", r2Data.error);
            throw new Error(r2Data.error);
        }
        
        // Call backend API for MSE calculation
        const mseResponse = await fetch('/api/calculate_mse', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                y_true: y_true,
                y_pred: y_pred
            })
        });
        
        const mseData = await mseResponse.json();
        if (!mseData.success) {
            console.error("Error calculating MSE:", mseData.error);
            throw new Error(mseData.error);
        }
        
        return { 
            r2: r2Data.r2, 
            mse: mseData.mse 
        };
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
    
    // Reset current model and predictions
    currentModel = null;
    trainPredictions = null;
    testPredictions = null;
    
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
        await updateChart(currentData);
        
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
        
        // Reset predictions
        trainPredictions = null;
        testPredictions = null;
        
        // Disable buttons
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

// Function to reset all metrics
function resetAllMetrics() {
    r2ScoreElement.textContent = '-';
    mseElement.textContent = '-';
    r2ScorePredTestElement.textContent = '-';
    msePredTestElement.textContent = '-';
    trainTimeElement.textContent = '-';
    iterationsElement.textContent = '-';
    costImprovementElement.textContent = '-';
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
        // Debug the metrics fields specifically
        console.log("Metrics fields:", {
            avgR2: cvResults.avgR2,
            avgMSE: cvResults.avgMSE,
            avg_training_time: cvResults.avg_training_time,
            avg_iterations: cvResults.avg_iterations,
            finalModel: cvResults.finalModel
        });
        
        // Additional detailed debugging for the optimizer info
        if (cvResults.finalModel) {
            console.log("Final model details:", {
                hasOptimizerInfo: !!cvResults.finalModel.optimizer_info,
                optimizerInfo: cvResults.finalModel.optimizer_info,
                iterations: cvResults.finalModel.optimizer_info?.iterations,
                costImprovement: cvResults.finalModel.optimizer_info?.cost_improvement
            });
        }
        
        // Store information about the model on the server
        if (cvResults.finalModel) {
            // Store model weights and bias but NOT override the server model
            currentModel.weights = cvResults.finalModel.weights;
            currentModel.bias = cvResults.finalModel.bias;
            currentModel.modelId = cvResults.finalModel.model_id; // Store the server-side model ID for reference
            console.log(`Model trained and stored on server (ID: ${currentModel.modelId})`);
        }
        
        // Update the UI with cross-validation results
        r2ScoreElement.textContent = cvResults.avgR2.toFixed(4);
        mseElement.textContent = cvResults.avgMSE.toFixed(2);
        
        // Update additional training metrics if available
        if (cvResults.avg_training_time !== undefined) {
            trainTimeElement.textContent = cvResults.avg_training_time.toFixed(3);
        }
        
        // Update iterations if available
        if (cvResults.avg_iterations !== undefined && cvResults.avg_iterations !== null) {
            iterationsElement.textContent = Math.round(cvResults.avg_iterations);
        } 
        // Fallback to final model iterations if avg_iterations is not available
        else if (cvResults.finalModel?.optimizer_info?.iterations !== undefined) {
            iterationsElement.textContent = Math.round(cvResults.finalModel.optimizer_info.iterations);
        }
        
        // Calculate and display cost improvement
        if (cvResults.finalModel?.optimizer_info) {
            const optimizerInfo = cvResults.finalModel.optimizer_info;
            
            // Log all optimizer info for debugging
            console.log("Optimizer info for cost calculation:", optimizerInfo);
            
            // Option 1: Use pre-calculated cost improvement if available
            if (optimizerInfo.cost_improvement !== undefined) {
                costImprovementElement.textContent = optimizerInfo.cost_improvement.toFixed(2);
                console.log("Using pre-calculated cost improvement:", optimizerInfo.cost_improvement);
            } 
            // Option 2: Calculate from initial and final costs if available
            else if (optimizerInfo.initial_cost !== undefined && optimizerInfo.final_cost !== undefined && optimizerInfo.initial_cost !== 0) {
                const initialCost = optimizerInfo.initial_cost;
                const finalCost = optimizerInfo.final_cost;
                const improvement = ((initialCost - finalCost) / initialCost) * 100;
                costImprovementElement.textContent = improvement.toFixed(2);
                console.log("Calculated cost improvement from initial/final:", improvement);
            }
            // Option 3: Calculate from cost history if available
            else if (optimizerInfo.cost_history && optimizerInfo.cost_history.length >= 2) {
                const initialCost = optimizerInfo.cost_history[0];
                const finalCost = optimizerInfo.cost_history[optimizerInfo.cost_history.length - 1];
                
                if (initialCost !== 0) {
                    const improvement = ((initialCost - finalCost) / initialCost) * 100;
                    costImprovementElement.textContent = improvement.toFixed(2);
                    console.log("Calculated cost improvement from history:", improvement);
                } else {
                    console.warn("Initial cost is zero, cannot calculate improvement percentage");
                }
            } else {
                console.warn("No suitable cost data found for calculating improvement");
            }
        } else {
            console.warn("No optimizer info available for cost improvement calculation");
        }
        
        // Reset prediction metrics for test data
        r2ScorePredTestElement.textContent = '-';
        msePredTestElement.textContent = '-';
        
        // Remove any highlighting
        const trainMetricsCard = r2ScoreElement.closest('.card');
        trainMetricsCard.style.border = '';
        trainMetricsCard.title = '';
        
        const testPredMetricsCard = r2ScorePredTestElement.closest('.card');
        testPredMetricsCard.style.border = '';
        testPredMetricsCard.title = '';

        // Set view mode to training data only
        viewMode = 'train';
        
        // Generate predictions ONLY for training data after model training
        try {
            trainButton.textContent = 'Generating training predictions...';
            
            // Only generate predictions for training data
            trainPredictions = await currentModel.predict(trainData.X);
            
            console.log("Successfully made predictions for training data");
            console.log(`Train predictions (first 5): ${trainPredictions.slice(0, 5)}`);
            
            // Reset test predictions
            testPredictions = null;
            
        } catch (predError) {
            console.error("Training prediction failed:", predError);
            // Don't throw, just log the error - we'll still show the trained model
        }
        
        // Update visualization to show only training predictions
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

// Function to highlight performance difference between training and prediction metrics
function highlightPerformanceDifference(trainR2, testR2) {
    const trainMetricsCard = r2ScoreElement.closest('.card');
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
        
        if (!currentModel.weights || currentModel.bias === null) {
            throw new Error('Model not properly trained. Please train the model again.');
        }

        console.log(`Making predictions with model (Server ID: ${currentModel.modelId || 'unknown'})`);
        
        // Make predictions on test dataset only using the server-side model
        try {
            // Set a loading state
            predictButton.disabled = true;
            predictButton.textContent = 'Making test predictions...';
            
            // Only generate predictions for test data
            testPredictions = await currentModel.predict(testData.X);
            
            console.log("Successfully made predictions for test data");
            console.log(`Test predictions (first 5): ${testPredictions.slice(0, 5)}`);
            console.log(`Test predictions length: ${testPredictions.length}`);
        } catch (predError) {
            console.error("Prediction failed:", predError);
            throw new Error(`Prediction failed: ${predError.message}. Try training the model again.`);
        } finally {
            // Reset button state
            predictButton.disabled = false;
            predictButton.textContent = 'Make Predictions';
        }
        
        // Calculate and display metrics for test predictions only
        const testPredMetrics = await calculateMetrics(testData.X, testData.y, testPredictions);
        console.log("Test data metrics:", testPredMetrics);
        
        r2ScorePredTestElement.textContent = testPredMetrics.r2.toFixed(4);
        msePredTestElement.textContent = testPredMetrics.mse.toFixed(2);
        
        // Highlight performance difference
        highlightPerformanceDifference(parseFloat(r2ScoreElement.textContent), testPredMetrics.r2);
        
        // Switch to test view mode to show the test predictions
        viewMode = 'test';
        
        // Update the visualization
        updateVisualization();
        
        // Update toggle button text
        if (testModeToggle) testModeToggle.textContent = 'View Both';
        
    } catch (error) {
        console.error("Prediction error:", error);
        alert('Error: ' + error.message);
    }
}); 