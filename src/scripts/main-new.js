// 简化的主JavaScript文件

// 应用初始化函数
function initApp() {
    console.log('App initialized');
    
    // 添加响应式导航菜单功能
    setupResponsiveNav();
    
    // 高亮当前导航项
    highlightActiveNavItem();
}

// 设置响应式导航菜单
function setupResponsiveNav() {
    const nav = document.querySelector('nav');
    const navLinks = document.querySelector('.nav-links');
    
    if (nav && navLinks) {
        const navToggle = document.createElement('button');
        navToggle.className = 'nav-toggle';
        navToggle.innerHTML = '&#9776;';
        nav.appendChild(navToggle);
        
        navToggle.addEventListener('click', function() {
            navLinks.classList.toggle('show');
        });
    }
}

// 高亮当前导航项
function highlightActiveNavItem() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(function(link) {
        const linkPath = link.getAttribute('href');
        if (currentPath.includes(linkPath) || (linkPath === '/' && currentPath === '/')) {
            link.classList.add('active');
        }
    });
}

// 等待DOM加载完成后初始化应用
document.addEventListener('DOMContentLoaded', function() {
    initApp();
});