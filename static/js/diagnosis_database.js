// diagnosis_database.js
// 诊断数据库页面前端交互脚本

document.addEventListener('DOMContentLoaded', function () {
    // 模拟数据
    const mockData = [
        {id: 1, patient_id: 'A001', gender: '男', age: 45, date: '2025-08-01', image: '/static/images/lungs-icon.svg'},
        {id: 2, patient_id: 'A002', gender: '女', age: 52, date: '2025-08-02', image: '/static/images/lungs-icon.svg'},
        {id: 3, patient_id: 'A003', gender: '男', age: 36, date: '2025-08-03', image: '/static/images/lungs-icon.svg'},
        {id: 4, patient_id: 'A004', gender: '女', age: 60, date: '2025-08-04', image: '/static/images/lungs-icon.svg'}
    ];

    // 填充数据表格
    function renderTable(data) {
        const tbody = document.getElementById('database-table-body');
        tbody.innerHTML = '';
        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.id}</td>
                <td>${row.patient_id}</td>
                <td>${row.gender}</td>
                <td>${row.age}</td>
                <td>${row.date}</td>
                <td><img src="${row.image}" alt="影像" class="img-thumbnail" style="width:48px;cursor:pointer;" data-img="${row.image}"></td>
                <td><button class="btn btn-sm btn-outline-primary view-btn" data-img="${row.image}">看片</button></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // 填充数据总量
    function renderCount(count) {
        document.getElementById('data-count').textContent = count;
    }

    // 渲染统计图表
    function renderChart(data) {
        const chartDom = document.getElementById('data-stats-chart');
        const myChart = echarts.init(chartDom);
        const option = {
            title: {text: '数据量统计', left: 'center', top: 10, textStyle: {fontSize: 16}},
            tooltip: {},
            xAxis: {type: 'category', data: data.map(d => d.date)},
            yAxis: {type: 'value'},
            series: [{
                data: data.map(d => 1),
                type: 'bar',
                itemStyle: {color: '#2B5DE0'}
            }]
        };
        myChart.setOption(option);
    }

    // 搜索功能
    document.getElementById('search-input').addEventListener('input', function () {
        const val = this.value.trim();
        const filtered = mockData.filter(row =>
            row.patient_id.includes(val) ||
            row.gender.includes(val) ||
            row.age.toString().includes(val) ||
            row.date.includes(val)
        );
        renderTable(filtered);
    });

    // 影像预览
    document.getElementById('database-table-body').addEventListener('click', function (e) {
        if (e.target.classList.contains('view-btn') || e.target.tagName === 'IMG') {
            const img = e.target.getAttribute('data-img');
            document.getElementById('preview-image').src = img;
            const modal = new bootstrap.Modal(document.getElementById('imageModal'));
            modal.show();
        }
    });

    // 导入按钮
    document.getElementById('import-btn').addEventListener('click', function () {
        document.getElementById('import-file-input').click();
    });

    // 导出按钮
    document.getElementById('export-btn').addEventListener('click', function () {
        alert('导出功能仅为前端演示，后端未实现。');
    });

    // 文件导入（仅演示）
    document.getElementById('import-file-input').addEventListener('change', function () {
        alert('导入功能仅为前端演示，后端未实现。');
        this.value = '';
    });

    // 初始化
    renderTable(mockData);
    renderCount(mockData.length);
    renderChart(mockData);
});

