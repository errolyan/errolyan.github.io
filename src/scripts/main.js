// 主JavaScript文件

// 应用初始化函数
function initApp() {
    console.log('App initialized');
    
    // 初始化markdown解析器（如果存在）
    if (window.markdownParser) {
        try {
            window.markdownParser.init();
        } catch (e) {
            console.error('Error initializing markdown parser:', e);
        }
    }
    
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
        navToggle.setAttribute('aria-label', '菜单');
        
        nav.appendChild(navToggle);
        
        navToggle.addEventListener('click', function() {
            navLinks.classList.toggle('show');
            navToggle.classList.toggle('active');
        });
    }
}

// 高亮当前导航项
function highlightActiveNavItem() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(function(link) {
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
    window.addEventListener('scroll', function() {
        const header = document.querySelector('header');
        if (header) {
            if (window.scrollY > 50) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
        }
    });
}

// 等待DOM加载完成后初始化应用
document.addEventListener('DOMContentLoaded', function() {
    initApp();
    setupScrollEffects();
});

// 导出函数供其他模块使用
window.initApp = initApp;