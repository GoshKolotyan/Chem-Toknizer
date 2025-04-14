# TODOs for Perovskite Tokenizer Project

## Data Analysis & Preparation
- [ ] Analyze dataset to identify most frequent element combinations for A, B, X sites
- [ ] Calculate statistical distributions of element proportions at each site
- [ ] Create visualization of element frequencies to guide tokenization strategy
- [ ] Identify outliers or problematic formulations in the dataset
- [ ] Determine optimal binning strategy for continuous values (radii, tolerance factor)

## Tokenizer Structure
- [ ] Refactor tokenizer to ensure fixed shape output tensor [1, 25]
- [ ] Define consistent positions for each feature type in the output tensor
- [ ] Create mapping documentation for tensor positions to feature meanings
- [ ] Implement robust error handling for edge cases in chemical formulas
- [ ] Add validation rules to ensure tokenizer completeness

## Charge Balance Implementation
- [ ] Fix charge calculation to ensure balance within [-0.01, 0.01] range
- [ ] Implement warning/flag for formulations outside acceptable charge range
- [ ] Add normalization for slight charge imbalances within tolerance
- [ ] Create test cases for charge calculation with known balanced materials
- [ ] Implement charge-balanced data generation for augmentation

## Element Properties & Encoding
- [ ] Collect comprehensive element property data for embeddings
- [ ] Create property vectors for each element (periodic table position, electronegativity, etc.)
- [ ] Develop encoding strategy for mixed-cation/anion perovskites
- [ ] Implement site-specific encoding for A, B, X elements
- [ ] Add position-aware information to element encodings

## Testing & Validation
- [ ] Create unit tests for each tokenizer component
- [ ] Build test suite for various perovskite formulations
- [ ] Implement validation for physical constraints (tolerance factor, etc.)
- [ ] Create visualization tools to inspect tokenizer outputs
- [ ] Set up benchmarks to evaluate tokenizer performance

## Integration with Model
- [ ] Ensure compatibility of tokenizer output with transformer input
- [ ] Create data pipeline from raw formulas to model-ready tensors
- [ ] Implement batch processing for efficient training
- [ ] Design feature extraction utility functions for the model
- [ ] Add debugging tools to track tokenization through the pipeline

## Documentation & Reporting
- [ ] Document tokenization strategy and physical principles
- [ ] Create examples showing tokenization process step-by-step
- [ ] Prepare visualization of element encoding scheme
- [ ] Write tutorial for adding new elements to the tokenizer
- [ ] Document tensor format for downstream tasks

## Optimization
- [ ] Profile tokenizer performance and identify bottlenecks
- [ ] Optimize regex patterns for faster tokenization
- [ ] Implement caching for common element combinations
- [ ] Reduce memory footprint for large batch processing
- [ ] Parallelize tokenization for multi-sample processing