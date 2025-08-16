// diagnosis_database.js
// 诊断数据库前端交互脚本

document.addEventListener('DOMContentLoaded', function() {
    // 1. 获取统计数据
    fetch('/diagnosis/api/database-stats/')
        .then(res => res.json())
        .then(data => {
            document.getElementById('total-cases').textContent = data.total_cases || '--';
            document.getElementById('total-images').textContent = data.total_images || '--';
            document.getElementById('category-count').textContent = data.category_count || '--';
            document.getElementById('latest-date').textContent = data.latest_date || '--';
        });

    // 2. 渲染病例趋势图
    fetch('/diagnosis/api/case-trend/')
        .then(res => res.json())
        .then(data => {
            const chart = echarts.init(document.getElementById('case-trend-chart'));
            chart.setOption({
                title: { text: '病例趋势', left: 'center', top: 10, textStyle: { fontSize: 16 } },
                tooltip: { trigger: 'axis' },
                xAxis: { type: 'category', data: data.dates },
                yAxis: { type: 'value' },
                series: [{ name: '病例数', type: 'line', data: data.counts, smooth: true, areaStyle: {} }]
            });
        });

    // 3. 渲染类别分布饼图
    fetch('/diagnosis/api/category-distribution/')
        .then(res => res.json())
        .then(data => {
            const chart = echarts.init(document.getElementById('category-pie-chart'));
            chart.setOption({
                title: { text: '类别分布', left: 'center', top: 10, textStyle: { fontSize: 16 } },
                tooltip: { trigger: 'item' },
                legend: { bottom: 0 },
                series: [{
                    name: '类别',
                    type: 'pie',
                    radius: '60%',
                    data: data,
                    emphasis: {
                        itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0,0,0,0.2)' }
                    }
                }]
            });
        });

    // 4. 渲染数据表格（模拟数据已在HTML中）
    // 5. 查看看片/超声
    document.querySelector('#database-table').addEventListener('click', function(e) {
        // 超声数据可视化
        if (e.target.closest('.view-ultrasound-btn')) {
            const file = e.target.closest('.view-ultrasound-btn').dataset.file;
            // 跳转到Django路由的可视化页面
            window.open(`/diagnosis/ultrasound_viewer/?file=${encodeURIComponent(file)}`, '_blank');
            return;
        }
        // 其他病例（如CT）
        if (e.target.closest('.view-btn')) {
            const caseId = e.target.closest('.view-btn').dataset.id;
            window.open(`/diagnosis/database/view/${caseId}/`, '_blank');
        }
    });

    // 6. 导入弹窗
    document.getElementById('import-btn').addEventListener('click', function() {
        const modal = new bootstrap.Modal(document.getElementById('importModal'));
        modal.show();
    });
    document.getElementById('import-form').addEventListener('submit', function(e) {
        e.preventDefault();
        const fileInput = document.getElementById('import-file');
        if (!fileInput.files.length) return;
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        fetch('/diagnosis/api/database-import/', {
            method: 'POST',
            body: formData
        }).then(res => res.json()).then(data => {
            if (data.success) {
                alert('导入成功');
                loadTable();
            } else {
                alert('导入失败: ' + (data.error || '未知错误'));
            }
        });
    });

    // 7. 导出
    document.getElementById('export-btn').addEventListener('click', function() {
        window.location.href = '/diagnosis/api/database-export/';
    });
});
