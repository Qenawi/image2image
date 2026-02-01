/**
 * Main JavaScript for Image to Image Generator
 */

// Constants
const STATUS_POLL_INTERVAL_MS = 4000;

// DOM Elements
const engineStatus = document.getElementById('engine-status');
const downloadAlert = document.getElementById('download-alert');
const downloadMessage = document.getElementById('download-message');

// Status polling
let statusPollInterval = null;

// Store current model availability status globally
let currentModelStatus = null;

async function checkEngineStatus() {
    try {
        const response = await fetch('/api/status/');
        const data = await response.json();

        // Store for global access
        currentModelStatus = data;

        const downloadedCount = data.downloaded_models ? data.downloaded_models.length : 0;

        if (downloadedCount === 0) {
            // No models downloaded at all - show error
            updateStatusBadge('No Models', 'danger');
            showDownloadAlert(
                `No models downloaded.\n\n` +
                `Download using Hugging Face CLI:\n` +
                `HF_HOME=${data.cache_dir} huggingface-cli download <model_id>`,
                'danger'
            );
            stopStatusPolling();
        } else if (data.loading && data.loading.length > 0) {
            updateStatusBadge('Loading Model...', 'info');
        } else if (data.generation && data.generation.is_generating) {
            // Show generation progress
            const gen = data.generation;
            const etaText = gen.eta_seconds ? ` (~${Math.ceil(gen.eta_seconds / 60)}m left)` : '';
            updateStatusBadge(`Generating ${gen.progress_percent}%${etaText}`, 'warning');
            updateGenerationProgress(gen);
        } else if (downloadedCount > 0) {
            updateStatusBadge(`${downloadedCount} Models Ready`, 'success');
            hideGenerationProgress();
            // Stop polling once models are ready and not generating
            stopStatusPolling();
        } else {
            updateStatusBadge('Initializing', 'info');
        }

        // Dispatch custom event for page-specific handlers
        document.dispatchEvent(new CustomEvent('modelStatusUpdated', {
            detail: data
        }));

        return data;
    } catch (error) {
        updateStatusBadge('Error', 'danger');
        console.error('Status check failed:', error);
        return null;
    }
}

function updateStatusBadge(text, type) {
    if (!engineStatus) return;

    const iconClass = {
        'success': 'bi-check-circle-fill',
        'warning': 'bi-arrow-repeat',
        'danger': 'bi-exclamation-triangle-fill',
        'info': 'bi-hourglass-split',
    }[type] || 'bi-circle';

    engineStatus.innerHTML = `
        <span class="badge bg-${type}">
            <i class="bi ${iconClass}"></i> ${text}
        </span>
    `;
}

function showDownloadAlert(message, type = 'warning') {
    if (!downloadAlert) return;
    downloadMessage.textContent = message;
    downloadAlert.className = `alert alert-${type} alert-dismissible mx-3 mt-3`;
}

function hideDownloadAlert() {
    if (!downloadAlert) return;
    downloadAlert.classList.add('d-none');
}

// Expose function globally for use in templates
window.showDownloadAlert = showDownloadAlert;

function startStatusPolling() {
    // Check immediately
    checkEngineStatus();

    // Then poll periodically
    statusPollInterval = setInterval(checkEngineStatus, STATUS_POLL_INTERVAL_MS);
}

function stopStatusPolling() {
    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }
}

// CSRF Token helper
function getCsrfToken() {
    const tokenElement = document.querySelector('[name=csrfmiddlewaretoken]');
    if (tokenElement) return tokenElement.value;

    const cookieMatch = document.cookie.match(/csrftoken=([^;]+)/);
    return cookieMatch ? cookieMatch[1] : '';
}

// Generic AJAX form submission
async function submitFormAjax(form, url, options = {}) {
    const formData = new FormData(form);
    const csrfToken = getCsrfToken();

    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': csrfToken,
            },
        });

        const data = await response.json();

        if (options.onSuccess && response.ok) {
            options.onSuccess(data);
        } else if (options.onError && !response.ok) {
            options.onError(data);
        }

        return data;
    } catch (error) {
        if (options.onError) {
            options.onError({ error: error.message });
        }
        throw error;
    }
}

// Image preview helper
function setupImagePreview(inputElement, previewElement, containerElement) {
    if (!inputElement || !previewElement) return;

    inputElement.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(event) {
                previewElement.src = event.target.result;
                if (containerElement) {
                    containerElement.classList.remove('d-none');
                }
            };
            reader.readAsDataURL(file);
        }
    });
}

// Confirmation dialog helper
function confirmAction(message) {
    return confirm(message);
}

// Toast notifications (Bootstrap)
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();

    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    toastContainer.insertAdjacentHTML('beforeend', toastHtml);

    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { autohide: true, delay: 5000 });
    toast.show();

    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    document.body.appendChild(container);
    return container;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Format duration
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(1)}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = (seconds % 60).toFixed(0);
    return `${minutes}m ${remainingSeconds}s`;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Start status polling
    startStatusPolling();

    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    stopStatusPolling();
});

// Generation progress UI
function updateGenerationProgress(progress) {
    const progressBar = document.getElementById('generation-progress-bar');
    const progressText = document.getElementById('generation-progress-text');
    const progressContainer = document.getElementById('generation-progress');

    if (progressBar) {
        progressBar.style.width = `${progress.progress_percent}%`;
        progressBar.setAttribute('aria-valuenow', progress.progress_percent);
    }

    if (progressText) {
        const etaText = progress.eta_seconds
            ? ` - About ${Math.ceil(progress.eta_seconds / 60)} min remaining`
            : '';
        progressText.textContent = `Step ${progress.current_step}/${progress.total_steps} (${progress.progress_percent}%)${etaText}`;
    }

    if (progressContainer) {
        progressContainer.classList.remove('d-none');
    }
}

function hideGenerationProgress() {
    const progressContainer = document.getElementById('generation-progress');
    if (progressContainer) {
        progressContainer.classList.add('d-none');
    }
}

// Start polling for generation progress
function startGenerationPolling() {
    if (!statusPollInterval) {
        statusPollInterval = setInterval(checkEngineStatus, 1000); // Poll every second during generation
    }
}

// Check if a specific model is available
function isModelAvailable(modelId) {
    if (!currentModelStatus || !currentModelStatus.downloaded_models) {
        return false;
    }
    return currentModelStatus.downloaded_models.includes(modelId);
}

// Get current model status
function getModelStatus() {
    return currentModelStatus;
}

// Expose functions globally
window.isModelAvailable = isModelAvailable;
window.getModelStatus = getModelStatus;
window.currentModelStatus = currentModelStatus;
window.startGenerationPolling = startGenerationPolling;
window.updateGenerationProgress = updateGenerationProgress;
window.hideGenerationProgress = hideGenerationProgress;
