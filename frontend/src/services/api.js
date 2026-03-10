const API_BASE_URL = 'http://localhost:5002';

export const classifyText = async (text, domain = "base") => {
    try {
        const response = await fetch(`${API_BASE_URL}/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input: text, domain: domain }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }

        return await response.json();
    } catch (error) {
        console.error("API Error:", error);
        throw error;
    }
};

export const checkHealth = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        return await response.json();
    } catch (error) {
        return { status: "down", error: error.message };
    }
};

export const uploadDataset = async (file, domainName) => {
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('domain_name', domainName);
        
        const response = await fetch(`${API_BASE_URL}/upload_dataset`, {
            method: 'POST',
            // Do not set Content-Type header manually for FormData
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || 'Network response was not ok');
        }

        return await response.json();
    } catch (error) {
        console.error("API Error during upload:", error);
        throw error;
    }
};
