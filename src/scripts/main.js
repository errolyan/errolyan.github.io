// 主JavaScript文件

// 应用初始化
function initApp() {
    // 等待DOM加载完成
    document.addEventListener('DOMContentLoaded', () => {
        console.log('App initialized');
        
        // 初始化markdown解析器
        if (window.markdownParser) {
            window.markdownParser.init();
        }
        
        // 添加响应式导航菜单功能
        setupResponsiveNav();
        
        // 高亮当前导航项
        highlightActiveNavItem();
    });
}

// 设置响应式导航菜单
function setupResponsiveNav() {
    const navToggle = document.createElement('button');
    navToggle.className = 'nav-toggle';
    navToggle.innerHTML = '☰';
    navToggle.setAttribute('aria-label', '菜单');
    
    const nav = document.querySelector('nav');
    const navLinks = document.querySelector('.nav-links');
    
    if (nav && navLinks) {
        nav.appendChild(navToggle);
        
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('show');
            navToggle.classList.toggle('active');
        });
    }
}

// 高亮当前导航项
function highlightActiveNavItem() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        
        // 处理根路径
        if (linkPath === '/' && currentPath === '/') {
            link.classList.add('active');
        } 
        // 处理其他路径
        else if (currentPath.includes(linkPath) && linkPath !== '/') {
            link.classList.add('active');
        }
    });
}

// 页面滚动效果
function setupScrollEffects() {
    // 监听滚动事件，添加导航栏样式变化
    window.addEventListener('scroll', () => {
        const header = document.querySelector('header');
        if (header && window.scrollY > 50) {
            header.classList.add('scrolled');
        } else if (header) {
            header.classList.remove('scrolled');
        }
    });
}

// 导出函数供其他模块使用
window.initApp = initApp;

// 自动初始化应用
initApp();
setupScrollEffects();