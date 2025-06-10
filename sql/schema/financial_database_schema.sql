-- Financial Analytics Database Schema
-- Credit Risk & Fraud Detection System
-- Author: Ram Bharat Chowdary Moturi
-- Date: 2025

-- Create database
CREATE DATABASE FinancialAnalytics;
USE FinancialAnalytics;

-- =====================================================
-- CUSTOMER MANAGEMENT TABLES
-- =====================================================

-- Customer master table
CREATE TABLE customers (
    customer_id INT PRIMARY KEY IDENTITY(1,1),
    first_name NVARCHAR(50) NOT NULL,
    last_name NVARCHAR(50) NOT NULL,
    email NVARCHAR(100) UNIQUE NOT NULL,
    phone NVARCHAR(20),
    date_of_birth DATE NOT NULL,
    ssn NVARCHAR(11) UNIQUE, -- Encrypted in production
    address_line1 NVARCHAR(100),
    address_line2 NVARCHAR(100),
    city NVARCHAR(50),
    state NVARCHAR(50),
    zip_code NVARCHAR(10),
    country NVARCHAR(50) DEFAULT 'USA',
    gender CHAR(1) CHECK (gender IN ('M', 'F')),
    marital_status NVARCHAR(20) CHECK (marital_status IN ('Single', 'Married', 'Divorced', 'Widowed')),
    education_level NVARCHAR(50),
    employment_status NVARCHAR(50),
    annual_income DECIMAL(15,2),
    employment_length INT, -- in months
    account_opened_date DATE NOT NULL DEFAULT GETDATE(),
    account_status NVARCHAR(20) DEFAULT 'Active' CHECK (account_status IN ('Active', 'Inactive', 'Suspended', 'Closed')),
    risk_rating NVARCHAR(10) CHECK (risk_rating IN ('Low', 'Medium', 'High')),
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);

-- Customer financial profile
CREATE TABLE customer_financial_profile (
    profile_id INT PRIMARY KEY IDENTITY(1,1),
    customer_id INT NOT NULL,
    credit_score INT CHECK (credit_score BETWEEN 300 AND 850),
    total_debt DECIMAL(15,2) DEFAULT 0,
    total_assets DECIMAL(15,2) DEFAULT 0,
    monthly_income DECIMAL(15,2),
    monthly_expenses DECIMAL(15,2),
    debt_to_income_ratio DECIMAL(5,4),
    credit_utilization_ratio DECIMAL(5,4),
    number_of_credit_accounts INT DEFAULT 0,
    total_credit_limit DECIMAL(15,2) DEFAULT 0,
    mortgage_balance DECIMAL(15,2) DEFAULT 0,
    investment_balance DECIMAL(15,2) DEFAULT 0,
    savings_balance DECIMAL(15,2) DEFAULT 0,
    profile_date DATE NOT NULL DEFAULT GETDATE(),
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE(),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- =====================================================
-- LOAN AND CREDIT TABLES
-- =====================================================

-- Loan applications
CREATE TABLE loan_applications (
    application_id INT PRIMARY KEY IDENTITY(1,1),
    customer_id INT NOT NULL,
    loan_type NVARCHAR(50) NOT NULL CHECK (loan_type IN ('Personal', 'Mortgage', 'Auto', 'Business', 'Student')),
    loan_purpose NVARCHAR(100),
    requested_amount DECIMAL(15,2) NOT NULL,
    loan_term_months INT NOT NULL,
    interest_rate DECIMAL(5,4),
    application_date DATE NOT NULL DEFAULT GETDATE(),
    application_status NVARCHAR(20) DEFAULT 'Pending' CHECK (application_status IN ('Pending', 'Approved', 'Rejected', 'Withdrawn')),
    approval_date DATE,
    rejection_reason NVARCHAR(500),
    risk_score DECIMAL(10,6),
    probability_of_default DECIMAL(10,6),
    loan_to_value_ratio DECIMAL(5,4),
    debt_to_income_after_loan DECIMAL(5,4),
    collateral_value DECIMAL(15,2),
    guarantor_id INT,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE(),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Active loans
CREATE TABLE loans (
    loan_id INT PRIMARY KEY IDENTITY(1,1),
    application_id INT NOT NULL,
    customer_id INT NOT NULL,
    loan_number NVARCHAR(20) UNIQUE NOT NULL,
    loan_type NVARCHAR(50) NOT NULL,
    principal_amount DECIMAL(15,2) NOT NULL,
    interest_rate DECIMAL(5,4) NOT NULL,
    loan_term_months INT NOT NULL,
    monthly_payment DECIMAL(10,2) NOT NULL,
    outstanding_balance DECIMAL(15,2) NOT NULL,
    disbursement_date DATE NOT NULL,
    first_payment_date DATE NOT NULL,
    maturity_date DATE NOT NULL,
    loan_status NVARCHAR(20) DEFAULT 'Active' CHECK (loan_status IN ('Active', 'Paid Off', 'Defaulted', 'Charged Off')),
    days_past_due INT DEFAULT 0,
    next_payment_date DATE,
    total_payments_made INT DEFAULT 0,
    total_interest_paid DECIMAL(15,2) DEFAULT 0,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE(),
    FOREIGN KEY (application_id) REFERENCES loan_applications(application_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Payment history
CREATE TABLE loan_payments (
    payment_id INT PRIMARY KEY IDENTITY(1,1),
    loan_id INT NOT NULL,
    payment_date DATE NOT NULL,
    due_date DATE NOT NULL,
    payment_amount DECIMAL(10,2) NOT NULL,
    principal_amount DECIMAL(10,2) NOT NULL,
    interest_amount DECIMAL(10,2) NOT NULL,
    late_fee DECIMAL(10,2) DEFAULT 0,
    payment_method NVARCHAR(50) CHECK (payment_method IN ('Bank Transfer', 'Check', 'Online', 'Auto Pay')),
    payment_status NVARCHAR(20) DEFAULT 'Completed' CHECK (payment_status IN ('Completed', 'Failed', 'Pending', 'Reversed')),
    days_late INT DEFAULT 0,
    outstanding_balance_after DECIMAL(15,2),
    created_at DATETIME2 DEFAULT GETDATE(),
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id)
);

-- =====================================================
-- TRANSACTION AND FRAUD DETECTION TABLES
-- =====================================================

-- Transaction records
CREATE TABLE transactions (
    transaction_id BIGINT PRIMARY KEY IDENTITY(1,1),
    customer_id INT NOT NULL,
    account_number NVARCHAR(20),
    transaction_date DATETIME2 NOT NULL DEFAULT GETDATE(),
    transaction_amount DECIMAL(15,2) NOT NULL,
    transaction_type NVARCHAR(50) NOT NULL CHECK (transaction_type IN ('Purchase', 'ATM Withdrawal', 'Transfer', 'Payment', 'Deposit')),
    merchant_name NVARCHAR(100),
    merchant_category NVARCHAR(50),
    merchant_location NVARCHAR(100),
    payment_method NVARCHAR(50) CHECK (payment_method IN ('Credit Card', 'Debit Card', 'Online', 'Mobile App')),
    card_number_last4 NVARCHAR(4),
    authorization_code NVARCHAR(20),
    transaction_status NVARCHAR(20) DEFAULT 'Completed' CHECK (transaction_status IN ('Completed', 'Pending', 'Failed', 'Reversed')),
    is_international BIT DEFAULT 0,
    currency_code NVARCHAR(3) DEFAULT 'USD',
    exchange_rate DECIMAL(10,6) DEFAULT 1.0,
    ip_address NVARCHAR(45),
    device_fingerprint NVARCHAR(100),
    geo_latitude DECIMAL(10,8),
    geo_longitude DECIMAL(11,8),
    created_at DATETIME2 DEFAULT GETDATE(),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Fraud detection results
CREATE TABLE fraud_detection_results (
    detection_id BIGINT PRIMARY KEY IDENTITY(1,1),
    transaction_id BIGINT NOT NULL,
    fraud_score DECIMAL(10,6) NOT NULL,
    risk_level NVARCHAR(10) CHECK (risk_level IN ('Low', 'Medium', 'High')),
    is_fraud_predicted BIT NOT NULL,
    model_version NVARCHAR(20),
    detection_rules_triggered NVARCHAR(MAX), -- JSON format
    manual_review_required BIT DEFAULT 0,
    manual_review_result NVARCHAR(20) CHECK (manual_review_result IN ('Confirmed Fraud', 'False Positive', 'Pending')),
    actual_fraud_flag BIT, -- Updated after investigation
    detection_timestamp DATETIME2 DEFAULT GETDATE(),
    reviewed_by NVARCHAR(50),
    review_timestamp DATETIME2,
    notes NVARCHAR(MAX),
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

-- Fraud alerts
CREATE TABLE fraud_alerts (
    alert_id BIGINT PRIMARY KEY IDENTITY(1,1),
    transaction_id BIGINT NOT NULL,
    detection_id BIGINT NOT NULL,
    alert_type NVARCHAR(50) NOT NULL,
    alert_priority NVARCHAR(10) CHECK (alert_priority IN ('Low', 'Medium', 'High', 'Critical')),
    alert_message NVARCHAR(500),
    alert_status NVARCHAR(20) DEFAULT 'Open' CHECK (alert_status IN ('Open', 'Investigating', 'Resolved', 'False Positive')),
    assigned_to NVARCHAR(50),
    created_at DATETIME2 DEFAULT GETDATE(),
    resolved_at DATETIME2,
    resolution_notes NVARCHAR(MAX),
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id),
    FOREIGN KEY (detection_id) REFERENCES fraud_detection_results(detection_id)
);

-- =====================================================
-- RISK MANAGEMENT TABLES
-- =====================================================

-- Credit risk assessments
CREATE TABLE credit_risk_assessments (
    assessment_id INT PRIMARY KEY IDENTITY(1,1),
    customer_id INT NOT NULL,
    assessment_date DATE NOT NULL DEFAULT GETDATE(),
    assessment_type NVARCHAR(50) CHECK (assessment_type IN ('Application', 'Periodic Review', 'Early Warning')),
    credit_score_at_assessment INT,
    probability_of_default DECIMAL(10,6),
    loss_given_default DECIMAL(10,6),
    exposure_at_default DECIMAL(15,2),
    expected_loss DECIMAL(15,2),
    risk_rating NVARCHAR(10) CHECK (risk_rating IN ('AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D')),
    risk_factors NVARCHAR(MAX), -- JSON format
    model_version NVARCHAR(20),
    assessed_by NVARCHAR(50),
    review_date DATE,
    notes NVARCHAR(MAX),
    created_at DATETIME2 DEFAULT GETDATE(),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Portfolio risk metrics
CREATE TABLE portfolio_risk_metrics (
    metric_id INT PRIMARY KEY IDENTITY(1,1),
    calculation_date DATE NOT NULL,
    total_exposure DECIMAL(20,2),
    total_expected_loss DECIMAL(20,2),
    var_95 DECIMAL(20,2), -- Value at Risk 95%
    var_99 DECIMAL(20,2), -- Value at Risk 99%
    expected_shortfall DECIMAL(20,2),
    concentration_risk DECIMAL(10,6),
    avg_probability_of_default DECIMAL(10,6),
    default_correlation DECIMAL(10,6),
    stress_test_loss DECIMAL(20,2),
    capital_requirement DECIMAL(20,2),
    created_at DATETIME2 DEFAULT GETDATE()
);

-- =====================================================
-- REGULATORY AND COMPLIANCE TABLES
-- =====================================================

-- Basel III regulatory capital
CREATE TABLE regulatory_capital (
    capital_id INT PRIMARY KEY IDENTITY(1,1),
    reporting_date DATE NOT NULL,
    tier1_capital DECIMAL(20,2),
    tier2_capital DECIMAL(20,2),
    total_capital DECIMAL(20,2),
    risk_weighted_assets DECIMAL(20,2),
    tier1_capital_ratio DECIMAL(10,6),
    total_capital_ratio DECIMAL(10,6),
    leverage_ratio DECIMAL(10,6),
    common_equity_tier1_ratio DECIMAL(10,6),
    capital_conservation_buffer DECIMAL(10,6),
    systemic_risk_buffer DECIMAL(10,6),
    meets_regulatory_requirements BIT,
    created_at DATETIME2 DEFAULT GETDATE()
);

-- Audit trail
CREATE TABLE audit_trail (
    audit_id BIGINT PRIMARY KEY IDENTITY(1,1),
    table_name NVARCHAR(50) NOT NULL,
    record_id BIGINT NOT NULL,
    action_type NVARCHAR(10) CHECK (action_type IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values NVARCHAR(MAX), -- JSON format
    new_values NVARCHAR(MAX), -- JSON format
    changed_by NVARCHAR(50) NOT NULL,
    change_timestamp DATETIME2 DEFAULT GETDATE(),
    change_reason NVARCHAR(200),
    ip_address NVARCHAR(45)
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Customer indexes
CREATE INDEX IX_customers_email ON customers(email);
CREATE INDEX IX_customers_account_status ON customers(account_status);
CREATE INDEX IX_customers_risk_rating ON customers(risk_rating);

-- Financial profile indexes
CREATE INDEX IX_financial_profile_customer_date ON customer_financial_profile(customer_id, profile_date);
CREATE INDEX IX_financial_profile_credit_score ON customer_financial_profile(credit_score);

-- Loan indexes
CREATE INDEX IX_loans_customer_status ON loans(customer_id, loan_status);
CREATE INDEX IX_loans_maturity_date ON loans(maturity_date);
CREATE INDEX IX_loans_days_past_due ON loans(days_past_due);

-- Transaction indexes
CREATE INDEX IX_transactions_customer_date ON transactions(customer_id, transaction_date);
CREATE INDEX IX_transactions_amount ON transactions(transaction_amount);
CREATE INDEX IX_transactions_merchant_category ON transactions(merchant_category);
CREATE INDEX IX_transactions_date ON transactions(transaction_date);

-- Fraud detection indexes
CREATE INDEX IX_fraud_detection_transaction ON fraud_detection_results(transaction_id);
CREATE INDEX IX_fraud_detection_score ON fraud_detection_results(fraud_score);
CREATE INDEX IX_fraud_detection_timestamp ON fraud_detection_results(detection_timestamp);

-- =====================================================
-- VIEWS FOR ANALYTICS
-- =====================================================

-- Customer 360 view
CREATE VIEW vw_customer_360 AS
SELECT 
    c.customer_id,
    c.first_name + ' ' + c.last_name AS full_name,
    c.email,
    c.annual_income,
    c.account_status,
    c.risk_rating,
    cfp.credit_score,
    cfp.debt_to_income_ratio,
    cfp.credit_utilization_ratio,
    COUNT(l.loan_id) AS active_loans,
    SUM(l.outstanding_balance) AS total_outstanding_debt,
    AVG(cr.probability_of_default) AS avg_default_probability,
    COUNT(t.transaction_id) AS total_transactions_last_year,
    SUM(t.transaction_amount) AS total_transaction_amount_last_year
FROM customers c
LEFT JOIN customer_financial_profile cfp ON c.customer_id = cfp.customer_id
    AND cfp.profile_date = (SELECT MAX(profile_date) FROM customer_financial_profile WHERE customer_id = c.customer_id)
LEFT JOIN loans l ON c.customer_id = l.customer_id AND l.loan_status = 'Active'
LEFT JOIN credit_risk_assessments cr ON c.customer_id = cr.customer_id
    AND cr.assessment_date = (SELECT MAX(assessment_date) FROM credit_risk_assessments WHERE customer_id = c.customer_id)
LEFT JOIN transactions t ON c.customer_id = t.customer_id 
    AND t.transaction_date >= DATEADD(YEAR, -1, GETDATE())
GROUP BY 
    c.customer_id, c.first_name, c.last_name, c.email, c.annual_income, 
    c.account_status, c.risk_rating, cfp.credit_score, cfp.debt_to_income_ratio, 
    cfp.credit_utilization_ratio;

-- Fraud monitoring dashboard view
CREATE VIEW vw_fraud_monitoring AS
SELECT 
    t.transaction_id,
    t.customer_id,
    t.transaction_date,
    t.transaction_amount,
    t.merchant_category,
    t.is_international,
    fdr.fraud_score,
    fdr.risk_level,
    fdr.is_fraud_predicted,
    fdr.manual_review_required,
    fa.alert_status,
    fa.alert_priority
FROM transactions t
INNER JOIN fraud_detection_results fdr ON t.transaction_id = fdr.transaction_id
LEFT JOIN fraud_alerts fa ON fdr.detection_id = fa.detection_id
WHERE t.transaction_date >= DATEADD(DAY, -30, GETDATE());

-- Portfolio risk summary view
CREATE VIEW vw_portfolio_risk_summary AS
SELECT 
    l.loan_type,
    COUNT(*) AS loan_count,
    SUM(l.outstanding_balance) AS total_exposure,
    AVG(cr.probability_of_default) AS avg_pd,
    SUM(l.outstanding_balance * cr.probability_of_default) AS expected_loss,
    COUNT(CASE WHEN l.days_past_due > 30 THEN 1 END) AS loans_past_due_30,
    COUNT(CASE WHEN l.days_past_due > 90 THEN 1 END) AS loans_past_due_90
FROM loans l
INNER JOIN credit_risk_assessments cr ON l.customer_id = cr.customer_id
    AND cr.assessment_date = (SELECT MAX(assessment_date) FROM credit_risk_assessments WHERE customer_id = l.customer_id)
WHERE l.loan_status = 'Active'
GROUP BY l.loan_type;

-- =====================================================
-- STORED PROCEDURES
-- =====================================================

-- Calculate customer risk score
CREATE PROCEDURE sp_calculate_customer_risk_score
    @customer_id INT,
    @risk_score DECIMAL(10,6) OUTPUT
AS
BEGIN
    DECLARE @credit_score INT, @debt_ratio DECIMAL(5,4), @utilization DECIMAL(5,4);
    DECLARE @income DECIMAL(15,2), @employment_length INT, @past_due_count INT;
    
    -- Get customer financial data
    SELECT 
        @credit_score = cfp.credit_score,
        @debt_ratio = cfp.debt_to_income_ratio,
        @utilization = cfp.credit_utilization_ratio,
        @income = c.annual_income,
        @employment_length = c.employment_length
    FROM customers c
    LEFT JOIN customer_financial_profile cfp ON c.customer_id = cfp.customer_id
        AND cfp.profile_date = (SELECT MAX(profile_date) FROM customer_financial_profile WHERE customer_id = c.customer_id)
    WHERE c.customer_id = @customer_id;
    
    -- Count past due loans
    SELECT @past_due_count = COUNT(*)
    FROM loans
    WHERE customer_id = @customer_id AND days_past_due > 30;
    
    -- Calculate risk score (simplified model)
    SET @risk_score = 
        CASE 
            WHEN @credit_score >= 750 THEN 0.02
            WHEN @credit_score >= 700 THEN 0.05
            WHEN @credit_score >= 650 THEN 0.10
            WHEN @credit_score >= 600 THEN 0.20
            ELSE 0.35
        END +
        CASE 
            WHEN @debt_ratio > 0.5 THEN 0.15
            WHEN @debt_ratio > 0.4 THEN 0.10
            WHEN @debt_ratio > 0.3 THEN 0.05
            ELSE 0.00
        END +
        CASE 
            WHEN @utilization > 0.9 THEN 0.10
            WHEN @utilization > 0.8 THEN 0.05
            ELSE 0.00
        END +
        (@past_due_count * 0.05);
    
    -- Cap at 1.0 (100% probability)
    SET @risk_score = CASE WHEN @risk_score > 1.0 THEN 1.0 ELSE @risk_score END;
END;

-- Generate daily risk report
CREATE PROCEDURE sp_generate_daily_risk_report
    @report_date DATE = NULL
AS
BEGIN
    IF @report_date IS NULL
        SET @report_date = GETDATE();
    
    -- Summary statistics
    SELECT 
        'Portfolio Summary' AS report_section,
        COUNT(DISTINCT customer_id) AS total_customers,
        COUNT(*) AS total_active_loans,
        SUM(outstanding_balance) AS total_exposure,
        AVG(days_past_due) AS avg_days_past_due,
        COUNT(CASE WHEN days_past_due > 30 THEN 1 END) AS loans_past_due_30,
        COUNT(CASE WHEN days_past_due > 90 THEN 1 END) AS loans_past_due_90
    FROM loans
    WHERE loan_status = 'Active';
    
    -- Risk distribution
    SELECT 
        'Risk Distribution' AS report_section,
        risk_rating,
        COUNT(*) AS customer_count,
        AVG(probability_of_default) AS avg_pd,
        SUM(exposure_at_default) AS total_exposure
    FROM credit_risk_assessments cr
    INNER JOIN customers c ON cr.customer_id = c.customer_id
    WHERE cr.assessment_date = (SELECT MAX(assessment_date) FROM credit_risk_assessments WHERE customer_id = cr.customer_id)
    GROUP BY risk_rating
    ORDER BY risk_rating;
    
    -- Fraud statistics
    SELECT 
        'Fraud Summary' AS report_section,
        COUNT(*) AS total_transactions,
        COUNT(CASE WHEN fdr.is_fraud_predicted = 1 THEN 1 END) AS predicted_fraud,
        AVG(fdr.fraud_score) AS avg_fraud_score,
        COUNT(CASE WHEN fa.alert_status = 'Open' THEN 1 END) AS open_alerts
    FROM transactions t
    LEFT JOIN fraud_detection_results fdr ON t.transaction_id = fdr.transaction_id
    LEFT JOIN fraud_alerts fa ON fdr.detection_id = fa.detection_id
    WHERE CAST(t.transaction_date AS DATE) = @report_date;
END;

-- =====================================================
-- TRIGGERS FOR AUDIT TRAIL
-- =====================================================

-- Audit trigger for customers table
CREATE TRIGGER tr_customers_audit
ON customers
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    
    IF EXISTS(SELECT * FROM inserted) AND EXISTS(SELECT * FROM deleted)
    BEGIN
        -- UPDATE
        INSERT INTO audit_trail (table_name, record_id, action_type, old_values, new_values, changed_by)
        SELECT 
            'customers',
            i.customer_id,
            'UPDATE',
            (SELECT * FROM deleted d WHERE d.customer_id = i.customer_id FOR JSON AUTO),
            (SELECT * FROM inserted i2 WHERE i2.customer_id = i.customer_id FOR JSON AUTO),
            SUSER_NAME()
        FROM inserted i;
    END
    ELSE IF EXISTS(SELECT * FROM inserted)
    BEGIN
        -- INSERT
        INSERT INTO audit_trail (table_name, record_id, action_type, new_values, changed_by)
        SELECT 
            'customers',
            customer_id,
            'INSERT',
            (SELECT * FROM inserted i WHERE i.customer_id = inserted.customer_id FOR JSON AUTO),
            SUSER_NAME()
        FROM inserted;
    END
    ELSE IF EXISTS(SELECT * FROM deleted)
    BEGIN
        -- DELETE
        INSERT INTO audit_trail (table_name, record_id, action_type, old_values, changed_by)
        SELECT 
            'customers',
            customer_id,
            'DELETE',
            (SELECT * FROM deleted d WHERE d.customer_id = deleted.customer_id FOR JSON AUTO),
            SUSER_NAME()
        FROM deleted;
    END
END;