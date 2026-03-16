// BRAND → MODEL
$("#brand").change(function () {
  let brand = $(this).val();

  $("#model").html("<option value=''>Select Model</option>");
  $("#year").html("<option value=''>Select Year</option>");
  $("#carinfo").html("");
  $("#price").html("");

  if (brand) {
    $.get("/get_models/" + brand, function (data) {
      data.forEach(function (model) {
        $("#model").append(`<option value="${model}">${model}</option>`);
      });
    });
  }
});

// MODEL → YEAR
$("#model").change(function () {
  let brand = $("#brand").val();
  let model = $(this).val();

  $("#year").html("<option value=''>Select Year</option>");
  $("#carinfo").html("");
  $("#price").html("");

  if (model) {
    $.get(`/get_years/${brand}/${model}`, function (data) {
      data.forEach(function (year) {
        $("#year").append(`<option value="${year}">${year}</option>`);
      });
    });
  }
});

// YEAR → SHOW CAR DATA
$("#year").change(function () {
  let brand = $("#brand").val();
  let model = $("#model").val();
  let year = $(this).val();

  if (year) {
    $.get(`/get_car_data/${brand}/${model}/${year}`, function (data) {
      let html = "";

      for (let key in data) {
        html += `
        <div class="car-item">
          <b>${key}</b>: ${data[key]}
        </div>
        `;
      }

      $("#carinfo").html(html);
    });
  }
});

// PRICE ANIMATION FUNCTION
function animatePrice(finalPrice) {
  let element = $("#price");
  let current = 0;

  const duration = 1200;
  const steps = 60;
  const increment = finalPrice / steps;

  const timer = setInterval(function () {
    current += increment;

    if (current >= finalPrice) {
      current = finalPrice;
      clearInterval(timer);
    }

    element.html("Predicted Price: $" + Math.round(current).toLocaleString());
  }, duration / steps);
}

// PREDICT PRICE
$("#predict").click(function () {
  let brand = $("#brand").val();
  let model = $("#model").val();
  let year = $("#year").val();

  if (!brand || !model || !year) {
    alert("Please select Brand, Model, and Year");
    return;
  }

  $.post(
    "/predict",
    { brand: brand, model: model, year: year },
    function (data) {
      if (data.error) {
        $("#price").html(data.error);
      } else {
        let price = data.predicted_price;

        let low = price * 0.9;
        let high = price * 1.1;

        $("#price").html(`
          <div class="price-title">Estimated Price</div>
          <div id="animatedPrice">$0</div>
          <div class="price-range">
            Range: $${Math.round(low).toLocaleString()} - $${Math.round(high).toLocaleString()}
          </div>
        `);

        animatePrice(price);
      }
    },
  );
});
