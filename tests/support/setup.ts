// Browser APIs jsdom leaves out, stubbed for the DOM tests.
if (typeof Element !== 'undefined' && !Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function scrollIntoView() {};
}
// jsdom does no layout: give SVG charts the size the popups draw them at, so pointer positions map to chart values.
if (typeof SVGElement !== 'undefined') {
  SVGElement.prototype.getBoundingClientRect = function getBoundingClientRect() {
    return { x: 0, y: 0, left: 0, top: 0, width: 560, height: 300, right: 560, bottom: 300, toJSON: () => ({}) } as DOMRect;
  };
}
