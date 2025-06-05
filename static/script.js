// Initiate the word count to 0
document.getElementById('textInput').addEventListener('input', function() {
    const text = this.value.trim();
    const wordCount = text ? text.split(/\s+/).length : 0;
    document.getElementById('wordCount').textContent = wordCount;
    if (wordCount > 500) {
      document.getElementsByClassName('number')[0].classList.add("text-red-600");
      document.getElementsByClassName('number')[1].classList.add("text-red-600");
    }else{
      document.getElementsByClassName('number')[0].classList.remove("text-red-600");
      document.getElementsByClassName('number')[1].classList.remove("text-red-600");
    }
});


function validateWordCount(event) {
  const wordCount = parseInt(document.getElementById('wordCount').textContent);
  const errorMessage = document.getElementById('errorMessage');
  
  if (wordCount > 50 || wordCount < 1) {
      event.preventDefault();
      errorMessage.classList.remove('hidden');
      return false;
  }
  errorMessage.classList.add('hidden');
  document.getElementById("predictForm").submit();
}




function updateFormAction() {
    const form = document.getElementById('predictForm');
    const selected = document.querySelector('input[name="explainMethod"]:checked').value;
    form.action = `/explain/${selected}`;
}


  function showSection(section) {
        const sentimentForm = document.getElementById('predictForm');
        const brainSection = document.getElementById('brainSection');
        const sentimentSection = document.getElementById('sentimentSection');

        if (section === 'sentiment') {
            sentimentSection.style.display = 'flex';
            brainSection.style.display = 'none';

            updateFormAction();

        } else if (section === 'brain') {
          sentimentSection.style.display = 'none';
          brainSection.style.display = 'flex';

          updateMRIFormAction();
        }
    }
  // function updateFormAction() {
  //       const method = document.querySelector('input[name="explainMethod"]:checked')?.value;
  //       const form = document.getElementById('predictForm');

  //       if (method === 'lime') {
  //           form.action = '/explain/lime';
  //       } else if (method === 'shap') {
  //           form.action = '/explain/shap';
  //       } else if (method === 'gradcam') {
  //           form.action = '/explain/gradcam';
  //       } else if (method === 'lime_image') {
  //           form.action = '/explain/image/lime/';
  //       }
  //   }

  function updateMRIFormAction() {
    const selected = document.querySelector('input[name="image_explain_method"]:checked').value;
    const form = document.getElementById('mriForm');
    form.action = `/explain/${selected}`;
}

  document.addEventListener('DOMContentLoaded', function () {
      showSection('sentiment');
  });
  
