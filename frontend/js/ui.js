// UI 交互逻辑
class LockDetectionUI {
    constructor() {
        this.currentImage = null;
        this.initializeEventListeners();
        this.loadInitialData();
    }

    // 初始化事件监听器
    initializeEventListeners() {
        // 文件上传相关
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const selectFileBtn = document.getElementById('selectFileBtn');

        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // 拖拽上传
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileSelect(e);
        });

        // 点击选择文件按钮
        selectFileBtn.addEventListener('click', () => {
            fileInput.click();
        });
    }

    // 处理文件选择
    handleFileSelect(event) {
        const files = event.target.files || event.dataTransfer.files;
        
        if (files.length === 0) return;

        const file = files[0];
        
        // 检查文件类型
        if (!file.type.startsWith('image/')) {
            this.showMessage('请选择图片文件！', 'error');
            return;
        }

        // 检查文件大小 (限制为10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showMessage('图片大小不能超过10MB！', 'error');
            return;
        }

        this.currentImage = file;
        this.showImagePreview(file);
    }

    // 显示图片预览
    showImagePreview(file) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const previewImage = document.getElementById('previewImage');
            const imagePreview = document.getElementById('imagePreview');
            
            previewImage.src = e.target.result;
            imagePreview.style.display = 'block';
            
            // 隐藏上传区域
            document.getElementById('uploadArea').style.display = 'none';
        };

        reader.readAsDataURL(file);
    }

    // 清除图片
    clearImage() {
        this.currentImage = null;
        document.getElementById('imagePreview').style.display = 'none';
        document.getElementById('uploadArea').style.display = 'block';
        document.getElementById('resultSection').style.display = 'none';
        document.getElementById('fileInput').value = '';
    }

    // 检测锁状态
    async detectLocks() {
        if (!this.currentImage) {
            this.showMessage('请先选择图片！', 'error');
            return;
        }

        this.showLoading(true);

        try {
            const response = await api.detectLocks(this.currentImage);
            
            if (response.success) {
                this.displayResults(response.data);
                this.showMessage('检测完成！', 'success');
            } else {
                this.showMessage(response.message || '检测失败', 'error');
            }
        } catch (error) {
            console.error('检测失败:', error);
            this.showMessage('检测失败，请检查网络连接', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    // 显示检测结果
    displayResults(data) {
        const resultSection = document.getElementById('resultSection');
        const resultImage = document.getElementById('resultImage');
        
        // 显示结果区域
        resultSection.style.display = 'block';
        
        // 设置结果图片
        if (data.image_base64) {
            // 将hex转换为base64
            const hexString = data.image_base64;
            const bytes = new Uint8Array(hexString.match(/[\da-f]{2}/gi).map(h => parseInt(h, 16)));
            const blob = new Blob([bytes], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            resultImage.src = url;
        } else {
            // 如果没有结果图片，显示原图
            const reader = new FileReader();
            reader.onload = (e) => {
                resultImage.src = e.target.result;
            };
            reader.readAsDataURL(this.currentImage);
        }

        // 获取检测结果
        const result = data.result || {};
        
        // 填充基本信息
        document.getElementById('detectionId').textContent = data.detection_id || '-';
        document.getElementById('detectionTime').textContent = result.detection_time ? new Date(result.detection_time).toLocaleString() : new Date().toLocaleString();
        document.getElementById('confidence').textContent = result.confidence_score ? `${(result.confidence_score * 100).toFixed(1)}%` : '-';

        // 填充状态统计
        const totalLocks = result.total_locks || 0;
        const unlockedLocks = result.unlocked_locks || 0;
        const lockedLocks = result.locked_locks || 0;

        document.getElementById('totalLocks').textContent = totalLocks;
        document.getElementById('unlockedLocks').textContent = unlockedLocks;
        document.getElementById('lockedLocks').textContent = lockedLocks;

        // 设置安全状态
        const safetyStatus = document.getElementById('safetyStatus');
        const safetyText = document.getElementById('safetyText');
        
        if (unlockedLocks === 0) {
            safetyStatus.className = 'safety-status safe';
            safetyText.textContent = '安全 - 所有锁已锁定';
        } else {
            safetyStatus.className = 'safety-status warning';
            safetyText.textContent = `警告 - ${unlockedLocks} 个锁未锁定`;
        }

        // 显示详细信息
        this.displayLockDetails(result.lock_details || []);

        // 滚动到结果区域
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    // 显示锁详细信息
    displayLockDetails(locks) {
        const lockDetails = document.getElementById('lockDetails');
        lockDetails.innerHTML = '';

        if (!locks || locks.length === 0) {
            lockDetails.innerHTML = '<div class="lock-detail-item"><div class="lock-detail-info"><div class="lock-detail-type">未检测到锁</div></div></div>';
            return;
        }

        locks.forEach((lock, index) => {
            const lockItem = document.createElement('div');
            const isLocked = lock.is_locked;
            const statusClass = isLocked ? 'locked' : 'unlocked';
            const statusText = isLocked ? '已锁定' : '未锁定';
            
            lockItem.className = `lock-detail-item ${statusClass}`;
            
            lockItem.innerHTML = `
                <div class="lock-detail-info">
                    <div class="lock-detail-type">${lock.lock_type || '未知类型'}</div>
                    <div class="lock-detail-status">${statusText}</div>
                </div>
                <div class="lock-detail-confidence">${(lock.confidence * 100).toFixed(1)}%</div>
            `;
            
            lockDetails.appendChild(lockItem);
        });
    }

    // 加载初始数据
    async loadInitialData() {
        try {
            await this.updateSystemStatus();
            await this.loadStats();
            await this.loadHistory();
        } catch (error) {
            console.error('加载初始数据失败:', error);
        }
    }

    // 更新系统状态
    async updateSystemStatus() {
        try {
            const response = await api.healthCheck();
            const statusIndicator = document.getElementById('systemStatus');
            
            if (response.success) {
                statusIndicator.className = 'status-indicator online';
                statusIndicator.querySelector('span').textContent = '系统正常';
            } else {
                statusIndicator.className = 'status-indicator offline';
                statusIndicator.querySelector('span').textContent = '系统异常';
            }
        } catch (error) {
            const statusIndicator = document.getElementById('systemStatus');
            statusIndicator.className = 'status-indicator offline';
            statusIndicator.querySelector('span').textContent = '系统离线';
        }
    }

    // 加载统计信息
    async loadStats() {
        try {
            const response = await api.getStats();
            
            console.log('统计API响应:', response);
            
            if (response.success && response.data) {
                // 处理嵌套的统计信息结构
                const detectionStats = response.data.detection_stats || {};
                const datasetStats = response.data.dataset_stats || {};
                
                console.log('检测统计:', detectionStats);
                console.log('数据集统计:', datasetStats);
                
                // 计算数据集总大小
                let totalDatasetSize = 0;
                if (datasetStats.train && datasetStats.train.total_images) {
                    totalDatasetSize += datasetStats.train.total_images;
                }
                if (datasetStats.val && datasetStats.val.total_images) {
                    totalDatasetSize += datasetStats.val.total_images;
                }
                if (datasetStats.test && datasetStats.test.total_images) {
                    totalDatasetSize += datasetStats.test.total_images;
                }
                
                // 计算安全检测次数
                const totalDetections = detectionStats.total_detections || 0;
                const unsafeDetections = detectionStats.unsafe_detections || 0;
                const safeDetections = totalDetections - unsafeDetections;
                
                console.log('计算后的统计:', {
                    totalDetections,
                    unsafeDetections,
                    safeDetections,
                    totalDatasetSize
                });
                
                document.getElementById('totalDetections').textContent = totalDetections || '-';
                document.getElementById('unlockedCount').textContent = detectionStats.total_unlocked || unsafeDetections || '-';
                document.getElementById('safeCount').textContent = safeDetections || '-';
                document.getElementById('datasetSize').textContent = totalDatasetSize || '-';
            } else {
                console.log('统计API返回无效数据:', response);
            }
        } catch (error) {
            console.error('加载统计信息失败:', error);
            // 设置默认值
            document.getElementById('totalDetections').textContent = '-';
            document.getElementById('unlockedCount').textContent = '-';
            document.getElementById('safeCount').textContent = '-';
            document.getElementById('datasetSize').textContent = '-';
        }
    }

    // 加载历史记录
    async loadHistory() {
        const limit = document.getElementById('limitSelect').value;
        const tableBody = document.getElementById('historyTableBody');
        
        // 显示加载状态
        tableBody.innerHTML = '<tr><td colspan="6" class="loading">加载中...</td></tr>';

        try {
            const response = await api.getHistory(limit);
            
            console.log('历史API响应:', response);
            
            if (response.success && response.data) {
                // 处理嵌套的历史记录结构
                const history = response.data.history || [];
                console.log('历史记录数据:', history);
                this.displayHistory(history);
            } else {
                console.log('历史API返回无效数据:', response);
                tableBody.innerHTML = '<tr><td colspan="6" class="loading">暂无数据</td></tr>';
            }
        } catch (error) {
            console.error('加载历史记录失败:', error);
            tableBody.innerHTML = '<tr><td colspan="6" class="loading">加载失败</td></tr>';
        }
    }

    // 显示历史记录
    displayHistory(history) {
        const tableBody = document.getElementById('historyTableBody');
        tableBody.innerHTML = '';

        if (!history || history.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6" class="loading">暂无数据</td></tr>';
            return;
        }

        history.forEach(record => {
            const row = document.createElement('tr');
            
            // 根据数据库字段映射获取数据
            const detectionId = record.id || record.detection_id || '-';
            const timestamp = record.detection_time || record.timestamp || '-';
            const totalLocks = record.locks_detected || record.total_locks || 0;
            const unlockedLocks = record.unlocked_locks || 0;
            const confidence = record.confidence_score || record.confidence || 0;
            
            const safetyStatus = unlockedLocks > 0 ? '警告' : '安全';
            const safetyClass = unlockedLocks > 0 ? 'warning' : 'safe';
            
            row.innerHTML = `
                <td>${detectionId}</td>
                <td>${timestamp ? new Date(timestamp).toLocaleString() : '-'}</td>
                <td>${totalLocks}</td>
                <td>${unlockedLocks}</td>
                <td><span class="safety-status ${safetyClass}">${safetyStatus}</span></td>
                <td>${confidence ? `${(confidence * 100).toFixed(1)}%` : '-'}</td>
            `;
            
            tableBody.appendChild(row);
        });
    }

    // 刷新统计信息
    async refreshStats() {
        this.showLoading(true);
        try {
            await this.loadStats();
            this.showMessage('统计信息已刷新', 'success');
        } catch (error) {
            this.showMessage('刷新失败', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    // 显示消息
    showMessage(message, type = 'info') {
        // 移除现有消息
        const existingMessage = document.querySelector('.message');
        if (existingMessage) {
            existingMessage.remove();
        }

        // 创建新消息
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const icon = type === 'success' ? 'check-circle' : 
                    type === 'error' ? 'exclamation-triangle' : 
                    'info-circle';
        
        messageDiv.innerHTML = `<i class="fas fa-${icon}"></i> ${message}`;
        
        // 插入到容器顶部
        const container = document.querySelector('.container');
        container.insertBefore(messageDiv, container.firstChild);
        
        // 3秒后自动消失
        setTimeout(() => {
            messageDiv.remove();
        }, 3000);
    }

    // 显示/隐藏加载状态
    showLoading(show) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (show) {
            loadingOverlay.classList.add('active');
        } else {
            loadingOverlay.classList.remove('active');
        }
    }

    // 显示API文档
    showApiDocs() {
        window.open('/docs', '_blank');
    }

    // 显示关于信息
    showAbout() {
        alert('智能锁检测系统\n\n基于 YOLOv8 和 FastAPI 构建\n支持锁状态分类检测\n\n版本: 1.0.0');
    }
}

// 全局函数
let ui;

// 初始化应用
function initializeApp() {
    ui = new LockDetectionUI();
}

// 全局函数（供HTML调用）
function detectLocks() {
    ui.detectLocks();
}

function clearImage() {
    ui.clearImage();
}

function refreshStats() {
    ui.refreshStats();
}

function loadHistory() {
    ui.loadHistory();
}

function showApiDocs() {
    ui.showApiDocs();
}

function showAbout() {
    ui.showAbout();
}