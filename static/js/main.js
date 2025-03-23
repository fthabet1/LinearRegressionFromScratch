// static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
    // Dark mode toggle functionality
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const icon = darkModeToggle.querySelector('i');
    
    // Function to update toggle button icon
    const updateIcon = (isDarkMode) => {
        if (isDarkMode) {
            icon.classList.remove('icon-sun-o');
            icon.classList.add('icon-moon-o');
        } else {
            icon.classList.remove('icon-moon-o');
            icon.classList.add('icon-sun-o');
        }
    };
    
    // Check for saved user preference
    const savedTheme = localStorage.getItem('theme');
    
    // If user has previously selected dark mode
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        updateIcon(true);
    }
    
    // Toggle dark mode on button click
    darkModeToggle.addEventListener('click', () => {
        const isDarkMode = document.body.classList.toggle('dark-mode');
        localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
        updateIcon(isDarkMode);
    });
    
    // Handle dataset choice radio buttons
    const sampleDatasetRadio = document.getElementById('sample_dataset');
    const customDatasetRadio = document.getElementById('custom_dataset');
    const sampleDatasetSelect = document.getElementById('sample_dataset_select');
    const fileUpload = document.getElementById('file_upload');
    
    sampleDatasetRadio.addEventListener('change', function() {
        if (this.checked) {
            sampleDatasetSelect.disabled = false;
            fileUpload.disabled = true;
        }
    });
    
    customDatasetRadio.addEventListener('change', function() {
        if (this.checked) {
            sampleDatasetSelect.disabled = true;
            fileUpload.disabled = false;
        }
    });
    
    // Handle model type selection and descriptions
    const modelTypeSelect = document.getElementById('model_type');
    const lambdaDiv = document.getElementById('lambda_div');
    const modelDescription = document.getElementById('model_description');
    
    const modelDescriptions = {
        'univariate': 'Simple linear regression with one independent variable.',
        'multivariate': 'Linear regression with multiple independent variables.',
        'lasso': 'Linear regression with L1 regularization for feature selection.',
        'ridge': 'Linear regression with L2 regularization to reduce overfitting.'
    };
    
    modelTypeSelect.addEventListener('change', function() {
        // Update regularization section visibility
        if (this.value === 'lasso' || this.value === 'ridge') {
            lambdaDiv.style.display = 'block';
        } else {
            lambdaDiv.style.display = 'none';
        }
        
        // Update model description text
        if (modelDescription) {
            modelDescription.textContent = modelDescriptions[this.value] || '';
        }
    });
    
    // Initialize lambda div visibility based on initial model select value
    if (modelTypeSelect) {
        if (modelTypeSelect.value === 'lasso' || modelTypeSelect.value === 'ridge') {
            lambdaDiv.style.display = 'block';
        } else {
            lambdaDiv.style.display = 'none';
        }
    }
    
    // Handle test size range slider
    const testSizeRange = document.getElementById('test_size');
    const testSizeValue = document.getElementById('test_size_value');
    
    if (testSizeRange && testSizeValue) {
        testSizeRange.addEventListener('input', function() {
            testSizeValue.textContent = this.value + '%';
        });
    }
    
    // Handle file upload and column selection
    const fileInput = document.getElementById('file_upload');
    const columnSelection = document.getElementById('column_selection');
    const targetColumnSelect = document.getElementById('target_column');
    const featureColumnsContainer = document.getElementById('feature_columns_container');
    
    if (fileInput && columnSelection && targetColumnSelect && featureColumnsContainer) {
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                const file = this.files[0];
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    const contents = e.target.result;
                    const lines = contents.split('\n');
                    if (lines.length > 0) {
                        // Assume first line contains headers
                        const headers = lines[0].split(',').map(header => header.trim());
                        
                        // Update target column select
                        targetColumnSelect.innerHTML = '';
                        headers.forEach(header => {
                            if (header) {
                                const option = document.createElement('option');
                                option.value = header;
                                option.textContent = header;
                                targetColumnSelect.appendChild(option);
                            }
                        });
                        
                        // Update feature columns checkboxes
                        featureColumnsContainer.innerHTML = '';
                        headers.forEach(header => {
                            if (header) {
                                const div = document.createElement('div');
                                div.className = 'form-group';
                                
                                const label = document.createElement('label');
                                label.className = 'form-checkbox feature-checkbox-label';
                                
                                const input = document.createElement('input');
                                input.type = 'checkbox';
                                input.name = 'feature_columns';
                                input.value = header;
                                
                                const i = document.createElement('i');
                                i.className = 'form-icon';
                                
                                const span = document.createElement('span');
                                span.textContent = header;
                                
                                label.appendChild(input);
                                label.appendChild(i);
                                label.appendChild(span);
                                div.appendChild(label);
                                featureColumnsContainer.appendChild(div);
                            }
                        });
                        
                        // Select last column as target by default
                        if (headers.length > 0) {
                            targetColumnSelect.value = headers[headers.length - 1];
                            
                            // Check all other columns as features by default
                            const checkboxes = featureColumnsContainer.querySelectorAll('input[type="checkbox"]');
                            checkboxes.forEach((checkbox, index) => {
                                if (index < headers.length - 1) {
                                    checkbox.checked = true;
                                }
                            });
                        }
                        
                        // Show column selection section
                        columnSelection.style.display = 'block';
                    }
                };
                
                reader.readAsText(file);
            }
        });
    }
    
    // Also handle sample dataset selection
    if (sampleDatasetSelect && columnSelection && targetColumnSelect && featureColumnsContainer) {
        // Sample placeholder columns for demo
        const sampleColumns = {
            'housing': ['area', 'bedrooms', 'bathrooms', 'age', 'price'],
            'advertising': ['tv', 'radio', 'newspaper', 'sales']
        };
        
        sampleDatasetSelect.addEventListener('change', function() {
            const selectedDataset = this.value;
            if (sampleColumns[selectedDataset]) {
                const columns = sampleColumns[selectedDataset];
                
                // Update target column select
                targetColumnSelect.innerHTML = '';
                columns.forEach(column => {
                    const option = document.createElement('option');
                    option.value = column;
                    option.textContent = column;
                    targetColumnSelect.appendChild(option);
                });
                
                // Default to last column as target (usually the dependent variable)
                targetColumnSelect.value = columns[columns.length - 1];
                
                // Update feature columns checkboxes
                featureColumnsContainer.innerHTML = '';
                columns.forEach(column => {
                    const div = document.createElement('div');
                    div.className = 'form-group';
                    
                    const label = document.createElement('label');
                    label.className = 'form-checkbox feature-checkbox-label';
                    
                    const input = document.createElement('input');
                    input.type = 'checkbox';
                    input.name = 'feature_columns';
                    input.value = column;
                    
                    const i = document.createElement('i');
                    i.className = 'form-icon';
                    
                    const span = document.createElement('span');
                    span.textContent = column;
                    
                    label.appendChild(input);
                    label.appendChild(i);
                    label.appendChild(span);
                    div.appendChild(label);
                    featureColumnsContainer.appendChild(div);
                    
                    // Check all columns except the target as features by default
                    if (column !== columns[columns.length - 1]) {
                        input.checked = true;
                    }
                });
                
                // Show column selection section
                columnSelection.style.display = 'block';
            }
        });
        
        // Trigger the change event to initialize column selection for the default dataset
        const event = new Event('change');
        sampleDatasetSelect.dispatchEvent(event);
    }

    // Show loading indicator on form submission
    const form = document.querySelector('form');
    const loadingIndicator = document.getElementById('loading_indicator');
    
    if (form && loadingIndicator) {
        form.addEventListener('submit', function() {
            loadingIndicator.style.display = 'block';
            this.querySelector('button[type="submit"]').disabled = true;
        });
    }
});