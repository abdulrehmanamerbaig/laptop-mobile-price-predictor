

const divs = document.querySelectorAll('.colors button');
divs.forEach(div => {
div.addEventListener('click', () => {
  const bgColorClass = div.className; // Get the class name of the clicked div
  document.getElementById('main-body').className = bgColorClass; // Set the class name to the body
  localStorage.setItem('selectedColor', bgColorClass);
});
});
const selectedColor = localStorage.getItem('selectedColor');

if (selectedColor) {
document.getElementById('main-body').className = selectedColor;
}