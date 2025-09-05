// API 接口封装
class LockDetectionAPI {
    constructor() {
        this.baseURL = 'http://localhost:8000';
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };
    }

    // 通用请求方法
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: { ...this.defaultHeaders, ...(options.headers || {}) },
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API请求失败:', error);
            throw error;
        }
    }

    // 健康检查
    async healthCheck() {
        return await this.request('/api/v1/health');
    }

    // 检测锁状态
    async detectLocks(file, userId = '') {
        const formData = new FormData();
        formData.append('file', file);
        
        const params = new URLSearchParams();
        if (userId) {
            params.append('user_id', userId);
        }

        const url = `/api/v1/lock/detect${params.toString() ? '?' + params.toString() : ''}`;
        
        return await this.request(url, {
            method: 'POST',
            body: formData,
            headers: {} // 让浏览器自动设置Content-Type
        });
    }

    // 获取系统统计信息
    async getStats() {
        return await this.request('/api/v1/stats');
    }

    // 获取检测历史
    async getHistory(limit = 10, offset = 0) {
        const params = new URLSearchParams();
        params.append('limit', limit);
        params.append('offset', offset);

        return await this.request(`/api/v1/history?${params.toString()}`);
    }

    // 配置钉钉机器人
    async configureDingtalk(config) {
        return await this.request('/api/v1/dingtalk/configure', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    // 图片对象检测（返回JSON）
    async detectObjectsJSON(file) {
        const formData = new FormData();
        formData.append('file', file);

        return await this.request('/img_object_detection_to_json', {
            method: 'POST',
            body: formData,
            headers: {}
        });
    }

    // 图片对象检测（返回图片）
    async detectObjectsImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseURL}/img_object_detection_to_img`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.blob();
    }
}

// 导出API实例
const api = new LockDetectionAPI();