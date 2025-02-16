const backToTopButton = document.getElementById('backToTop');

// Show the button when scrolled 100px down
window.addEventListener('scroll', () => {
    if (window.scrollY > 100) {
        backToTopButton.style.opacity = '1';
        backToTopButton.style.pointerEvents = 'auto';  // Enable clicking
    } else {
        backToTopButton.style.opacity = '0';
        backToTopButton.style.pointerEvents = 'none';  // Disable clicking
    }
});

// Scroll back to the top when clicked
backToTopButton.addEventListener('click', () => {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});
